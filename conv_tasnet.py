# author jax500 source code from Kai_tuo Xu
# Todo: parameters don't have name ~~ however not important

import tensorflow as tf
from tensorflow.python.keras import backend as k
eps = 1e-5


class GlobalLayerNorm(tf.keras.Model):
    def __init__(self, channel_size):
        """
        channel_last implementation of GLN
        GLN => LN compute reduce_mean[C, HW]
        :param channel_size:
        """
        super(GlobalLayerNorm, self).__init__()
        self.gamma = tf.Variable(tf.ones([1, 1, channel_size]), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, channel_size]), trainable=True)

    def call(self, inputs, training=None, mask=None):
        mean = k.mean(inputs, axis=[1, 2], keepdims=True)
        var = k.mean((inputs - mean)**2, axis=[1, 2], keepdims=True)
        return self.gamma * (inputs - mean) / tf.sqrt(var + eps) + self.beta


class ChannelWiseLayerNorm(tf.keras.Model):
    """
    normal LN: compute mean[C,HW]
    """
    def __init__(self, channel_size):
        super(ChannelWiseLayerNorm, self).__init__()
        self.gamma = tf.Variable(tf.ones([1, 1, channel_size]), trainable=True)
        self.beta = tf.Variable(tf.zeros([1, 1, channel_size]), trainable=True)

    def call(self, inputs, training=None, mask=None):
        mean = k.mean(inputs, axis=-1, keepdims=True)
        var = k.mean((inputs - mean)**2, axis=-1, keepdims=True)
        return self.gamma * (inputs - mean) / tf.sqrt(var + eps) + self.beta


def chose_norm(norm_type, channel_size):
    """
    The input of normalization will be (M, C, K), where M is batch size,
    C is channel size and K is sequence length.
    """
    norm_type = norm_type.lower()
    assert norm_type in ["gln", "cln", "bn"], "Typo norm type should be one of "
    if norm_type == "glb":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cln":
        return ChannelWiseLayerNorm(channel_size)
    else:
        # batch work at last dim such that [bs, hw] was reduce => [(bs, hw), c], repara at c
        return tf.keras.layers.BatchNormalization(axis=-1)


class Encoder(tf.keras.Model):
    """
    Org style: tf.nn.conv1D(tf.signal.frame(inputs, frame_length, frame_length//2))|kernel=frame_length
    1D conv Encoder map [bs, frame_index, time]2[bs, frame_index, basis]
    """
    def __init__(self, filters, kernel_size, enc_activation="relu"):
        super(Encoder, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=kernel_size//2,
                                             padding="valid",
                                             use_bias=False,
                                             activation=enc_activation)

    def call(self, inputs, training=None, mask=None):
        """
        In this work .5 overlap was build by stride conv and valid padding
        :param inputs: mixture input [bs, samples], M is batch size
        :param training:None
        :param mask :None
        :return: [bs, frame, filters] frame = int(samples - kernel_size)/(kernel_size//2) + 1
        = 2samples/kernel_size -1 make .5 overlap, frame_length = kernel_size
        """
        inputs_weights = self.conv1d(inputs)
        return inputs_weights


class Decoder(tf.keras.Model):
    """
    Decoder : Use to reform speech signal from basis, and overlap
    """
    def __init__(self, kernel_size_enc, activation=None, speech_length=32700):
        super(Decoder, self).__init__()
        self.audio_length = speech_length
        self.kernel_size_enc = kernel_size_enc
        self.dense = tf.keras.layers.Dense(units=kernel_size_enc,
                                           activation=activation,
                                           use_bias=False)

    def call(self, inputs, training=None, mask=None):
        """
        use overlap and add to reconstruct
        [bs, frame, basis] * [bs, frame, basis, spk]=>
        [bs, frame, basis, spk]=>[bs*spk, frame, basis]=dense>[bs*spk, frame, frame_length]
        =overlap and add>[bs*spk, samples]=>[bs, samples, spk]
        :param inputs: [bs, frame, chs]
        :param mask:[bs, frame, chs, spk]
        :param training:None
        :return: [bs, samples, spks]
        """

        inputs, estmask = inputs[0], inputs[1]
        _, frame, basis, spk = estmask.shape
        inputs = inputs[..., tf.newaxis] * estmask
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        inputs = tf.reshape(inputs, [-1, frame, basis])
        inputs = self.dense(inputs)
        inputs = tf.signal.overlap_and_add(inputs, frame_step=self.kernel_size_enc//2)
        inputs = tf.reshape(inputs, [-1, spk, self.audio_length])
        return inputs


class DepthwiseSeparableConv1D(tf.keras.Model):
    """
    implementation of DSC1D by DSC2D
    """
    def __init__(self,
                 input_chs,
                 out_channels,
                 kernel_size,
                 dilation,
                 norm_type="bn",
                 causal=False):
        """
        causal padding:[(k-1)*dilation, 0] padding
        normal padding:[(k-1)*dilation//2, (k-1)*dilation//2]

        :param input_chs:last layer's output channel used to normalize
        :param out_channels: output of 1x1 conv
        :param kernel_size: kernel of deep_wise conv
        :param dilation: dilation for deep wise conv
        :param norm_type: select_norm
        :param causal: whether use causal padding
        """
        assert norm_type in ["bn", "gLN", "cLN"]
        super(DepthwiseSeparableConv1D, self).__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal

        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D((kernel_size, 1),
                                                              dilation_rate=dilation,
                                                              activation=None,
                                                              use_bias=False,
                                                              padding="valid" if self.causal else "same")
        self.conv1x1 = tf.keras.layers.Conv1D(filters=out_channels,
                                              kernel_size=1,
                                              padding="same")
        self.norm = chose_norm(norm_type, channel_size=input_chs)

    def call(self, inputs, training=None, mask=None):
        inputs = inputs[:, :, tf.newaxis, :]
        if self.causal:
            inputs = tf.pad(inputs,
                            ((0, 0), ((self.kernel_size - 1) * self.dilation, 0), (0, 0), (0, 0)),
                            mode="REFLECT",
                            name=None)

        inputs = self.depthwise_conv(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        inputs = self.norm(inputs[:, :, 0, :], training)
        inputs = self.conv1x1(inputs)
        return inputs


class TemporalBlock(tf.keras.Model):
    """
    TB include conv1x1=>norm=>DSC1D
                      =>res =>      output
    """
    def __init__(self,
                 input_chs,
                 output_chs,
                 kernel_size,
                 dilation,
                 norm_type="gLN",
                 causal=False):
        super(TemporalBlock, self).__init__()
        assert norm_type.lower() in ["bn", "gln", "cln"]
        self.conv1x1 = tf.keras.layers.Conv1D(output_chs, 1, use_bias=False, padding="same")
        self.norm = chose_norm(norm_type, output_chs)
        self.norm_type = norm_type
        self.dsc = DepthwiseSeparableConv1D(output_chs,
                                            input_chs,
                                            kernel_size,
                                            dilation,
                                            norm_type="bn",
                                            causal=causal)

    def call(self, inputs, training=None, mask=None):
        res = inputs
        inputs = self.conv1x1(inputs)
        inputs = self.norm(inputs, training)
        inputs = tf.nn.leaky_relu(inputs)
        inputs = self.dsc(inputs)
        return inputs + res


class Padding(tf.keras.Model):
    """
    padding method layer, maybe not use
    """
    def __init__(self, padding, causal, mode):
        super(Padding, self).__init__()
        self.padding = padding
        self.causal = causal
        self.mode = mode

    def call(self, inputs, training=None, mask=None):
        if self.causal:
            inputs = tf.pad(inputs, ((0, 0), (self.padding, 0), (0, 0), (0, 0)), mode="constant", constant_values=0)
        else:
            length = inputs.shape[1]
            inputs = tf.pad(inputs, ((0, 0), (self.padding//2, self.padding//2+1), (0, 0), (0, 0)),
                            mode="constant", constant_values=0)
            inputs = inputs[:length]
        return inputs


class TemporlaConvNet(tf.keras.Model):
    """TCN as separator
       include: layer norm, bottleneck, repeat_outer_layer(repeat_TCN_block(TCNBlock)), maskconv with
       activation => mask non_linear
    """
    def __init__(self,
                 filters_ae,
                 filters_bottle,
                 filters_block,
                 kernel_size_block,
                 num_conv_block,
                 number_repeat,
                 spk_num,
                 norm_type="gLN",
                 causal=False,
                 mask_nonlinear='relu'):
        """

        :param filters_ae: encoder output chs
        :param filters_bottle: filters for bottleneck
        :param filters_block: filters for each T block
        :param kernel_size_block: .same
        :param num_conv_block: number of each repeat
        :param number_repeat: number of repeats
        :param spk_num: source number
        :param norm_type: ~
        :param causal: padding method
        :param mask_nonlinear: output activation
        """
        super(TemporlaConvNet, self).__init__()
        self.spk = spk_num
        self.norm_type = norm_type
        self.mask_nonlinear = mask_nonlinear
        if self.mask_nonlinear == "relu":
            self.last_activation = tf.keras.layers.Softmax()
        else:
            self.last_activation = tf.keras.layers.ReLU(negative_slope=.1)
        self.layer_norm = ChannelWiseLayerNorm(filters_ae)
        self.bottleneck_conv1x1 = tf.keras.layers.Conv1D(filters=filters_bottle,
                                                         kernel_size=1,
                                                         use_bias=False)
        self.model_repeat = tf.keras.Sequential()
        for r in range(number_repeat):
            for x in range(num_conv_block):
                dilation_rate = 2**x
                local_layer = TemporalBlock(filters_bottle, filters_block, kernel_size_block, dilation_rate,
                                            norm_type=norm_type, causal=causal)
                self.model_repeat.add(local_layer)
        self.mask_conv1x1 = tf.keras.layers.Conv1D(filters=spk_num*filters_ae,
                                                   kernel_size=1,
                                                   use_bias=False)

    def call(self, inputs, training=None, mask=None):
        inputs = self.layer_norm(inputs)
        inputs = self.bottleneck_conv1x1(inputs)
        inputs = self.model_repeat(inputs, training)
        inputs = self.mask_conv1x1(inputs)
        inputs = self.last_activation(inputs)
        inputs = tf.split(inputs, self.spk, axis=-1)
        inputs = tf.stack(inputs, axis=-1)
        return inputs


class ConvTasNet(tf.keras.Model):
    """
    Conv TaSNet follow work of KaiTuo Xu in TF 2.x
    """
    def __init__(self,
                 filters_e,
                 kernel_size_e,
                 bottle_filter,
                 filters_block,
                 kernel_size_block,
                 num_conv_block,
                 number_repeat,
                 spk_num,
                 norm_type="gLN",
                 causal=False,
                 mask_nonlinear='relu',
                 speech_length=32700,):
        """
        input_signals[bs, T, chs] =enc>
        mixtureweights[bs, frame, basis]=decoder>=mask estimate[bs, frame, basis, spk]>output
        It is noteworthy that kernel_size_e = frame_length
        :param filters_e: Number of filters in encoder
        :param kernel_size_e:Length of the filters in E
        :param bottle_filter:Number of channels in bottleneck 1 × 1-conv block
        :param filters_block:Number of channels in convolutional blocks
        :param kernel_size_block:Kernel size in convolutional blocks
        :param num_conv_block:Number of convolutional blocks in each repeat
        :param number_repeat:Number of repeats
        :param spk_num:Number of speakers
        :param norm_type:BN, gLN, cLN
        :param causal:causal or non-causal
        :param mask_nonlinear:use which non-linear function to generate mask in str
        :param speech_length: padded speech length
        """
        super(ConvTasNet, self).__init__()
        self.encoder = Encoder(filters_e, kernel_size_e)
        self.decoder = Decoder(kernel_size_e, speech_length=speech_length)
        self.separator = TemporlaConvNet(filters_e, bottle_filter, filters_block, kernel_size_block, num_conv_block,
                                         number_repeat, spk_num, norm_type, causal, mask_nonlinear)

    def call(self, inputs, training=None, mask=None):
        mixture_w = self.encoder(inputs=inputs, training=training)
        est_mask = self.separator(inputs=mixture_w, training=training)
        est_source = self.decoder(inputs=[mixture_w, est_mask], training=training)
        return est_source


if __name__ == '__main__':

    tf.enable_eager_execution()
    test_input = tf.ones([3, 1000, 1])
    test_input2 = tf.zeros([3, 1000, 1])
    test_input = tf.concat([test_input, test_input2], axis=-1)
    a = ConvTasNet(filters_e=10, kernel_size_e=50, bottle_filter=50, filters_block=50,
                   kernel_size_block=3, num_conv_block=4, number_repeat=2, spk_num=2,
                   speech_length=100)
    out = a(test_input)
    a.summary()
    print(out.shape)
