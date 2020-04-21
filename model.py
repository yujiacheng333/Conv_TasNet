import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as k
eps = 1e-6


class GLN(tf.keras.Model):
    def __init__(self):
        super(GLN, self).__init__()
        self.ndim = None
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        chs = input_shape[-1]
        self.ndim = len(input_shape)
        if self.ndim == 4:

            self.gamma = self.add_variable(name="gamma",
                                           shape=[1, 1, 1, chs],
                                           initializer=tf.keras.initializers.Ones(),
                                           trainable=True)
            self.beta = self.add_variable(name="beta",
                                          shape=[1, 1, 1, chs],
                                          initializer=tf.keras.initializers.Zeros(),
                                          trainable=True)
        if self.ndim == 3:
            self.gamma = self.add_variable(name="gamma",
                                           shape=[1, 1, chs],
                                           initializer=tf.keras.initializers.Ones(),
                                           trainable=True)
            self.beta = self.add_variable(name="beta",
                                          shape=[1, 1, chs],
                                          initializer=tf.keras.initializers.Zeros(),
                                          trainable=True)

    def call(self, inputs, training=None, mask=None):
        if self.ndim == 4:
            target_axis = [1, 2, 3]
        if self.ndim == 3:
            target_axis = [1, 2]
        mean, var = tf.nn.moments(inputs, keep_dims=True, axes=target_axis)
        inputs = (inputs - mean) / k.sqrt(var + eps) * self.gamma + self.beta
        return inputs


class Encoder(tf.keras.Model):
    def __init__(self, kernel_size, out_channels=64):
        super(Encoder, self).__init__()
        self.conv = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=out_channels, use_bias=False,
                                           strides=kernel_size//2, padding="same")

    def call(self, inputs, training=None, mask=None):
        """
        input_shape = [Bs, samplepts, chs~1 or not]
        """
        inputs = tf.nn.relu(self.conv(inputs))
        return inputs


class Decoder(tf.keras.Model):
    def __init__(self, kernel_size=2, num_speakers=1):
        super(Decoder, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(kernel_size=(kernel_size, 1), filters=num_speakers, use_bias=False,
                                                    strides=(kernel_size//2, 1), padding="same")

    def call(self, inputs, training=None, mask=None):
        """
        input_shape = [Bs, samplepts//stride, channels]
        output_shape = [Bs, samplepts, spk]
        """
        inputs = inputs[:, :, tf.newaxis, :]
        inputs = tf.nn.tanh(self.conv(inputs))
        return tf.squeeze(inputs)


def segmentation(x, k):
    """
    就是说填充了一个步长P之后，整个数据能够被划分为多少帧，如果还有多的，再填一帧
    双向填充，使得重叠相加更简单原始的这个填充大小不知道啥情况。。。应该只需要一个%
    :param x:[Bs, Lenght, rep_filters]
    :param k: frame_lenght, p:frame step
    :return: [Bs, N, K, chs] N = ((L + K//2) / k).upper * 2
    """
    bs, length, chs = x.get_shape().as_list()
    p = k // 2
    gap = k - (p + length) % k

    x = tf.pad(x, ((0, 0), (0+p, gap+p), (0, 0)))
    x_leading = x[:, :-p, :]
    x_leading = tf.reshape(x_leading, [bs, -1, k, chs])
    x_lagging = x[:, p:, :]
    x_lagging = tf.reshape(x_lagging, [bs, -1, k, chs])
    concate = tf.concat([x_leading, x_lagging], axis=-2)
    concate = tf.reshape(concate, [bs, -1, k, chs])
    return concate, gap


def over_add(x, gap):
    """

    :param x: [Bs, N, K, chs]
    :param gap: pad_length not include frame pad
    :return: seq_feature [Bs, samplepts, chs]
    """
    Bs, N, K, chs = x.get_shape().as_list()
    P = K//2
    x = tf.reshape(x, [Bs, -1, K*2, chs])
    x = tf.split(x, 2, axis=2)
    x_leadding = x[0]
    x_lagging = x[1]
    x_leadding = tf.reshape(x_leadding, [Bs, -1, chs])
    x_leadding = x_leadding[:, P:, :]
    x_lagging = tf.reshape(x_lagging, [Bs, -1, chs])
    x_lagging = x_lagging[:, :-P, :]
    recon = (x_leadding + x_lagging)[:, :-gap, :]
    return recon / 2


class DPRNNblock(tf.keras.Model):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 dropout=0):
        super(DPRNNblock, self).__init__()
        self.intra_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=hidden_channels, dropout=dropout, return_sequences=True))
        self.inter_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=hidden_channels, dropout=dropout, return_sequences=True))
        self.intra_norm = GLN()
        self.inter_norm = GLN()
        self.intra_linear = tf.keras.layers.Dense(units=out_channels, use_bias=False)
        self.inter_linear = tf.keras.layers.Dense(units=out_channels, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        """
        input_shape = [Bs, N, K, chs]
        eq shape mapping
        """
        res_0 = inputs
        Bs, N, K, chs = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [Bs*N, K, chs])
        inputs = self.intra_rnn(inputs)
        inputs = self.intra_linear(inputs)
        inputs = tf.reshape(inputs, [Bs, N, K, chs])
        inputs = self.intra_norm(inputs)
        inputs += res_0
        res_1 = inputs

        inputs = tf.reshape(tf.transpose(inputs, [0, 2, 1, 3]), [Bs*K, N, chs])
        inputs = self.inter_rnn(inputs)
        inputs = self.inter_linear(inputs)
        inputs = self.inter_norm(inputs)
        inputs = tf.transpose(tf.reshape(inputs, [Bs, K, N, chs]), [0, 2, 1, 3])
        inputs += res_1
        return inputs


class DPRNNmodel(tf.keras.Model):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 dropout=0, num_layers=4, K=200, num_spks=2):
        super(DPRNNmodel, self).__init__()
        self.K = K
        self.num_spk = num_spks
        self.norm = GLN()
        self.conv1d = tf.keras.layers.Conv1D(filters=out_channels, kernel_size=1, use_bias=False)
        self.dual_rnn = []
        for i in range(num_layers):
            self.dual_rnn.append(DPRNNblock(out_channels=out_channels,
                                            hidden_channels=hidden_channels,
                                            dropout=dropout))
        self.dual_rnn = tf.keras.Sequential(self.dual_rnn)
        self.conv2d = tf.keras.layers.Conv2D(filters=out_channels*self.num_spk, kernel_size=1)
        self.end_conv1x1 = tf.keras.layers.Conv1D(filters=in_channels, kernel_size=1, use_bias=False)
        self.Prelu = tf.keras.layers.PReLU()
        self.outputconv = tf.keras.layers.Conv1D(filters=out_channels, kernel_size=1, activation=tf.nn.tanh)
        self.outputgate = tf.keras.layers.Conv1D(filters=out_channels, kernel_size=1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        inputs = self.norm(inputs)
        inputs = self.conv1d(inputs)
        inputs, gap = segmentation(inputs, self.K)
        # [Bs, N, K, chs]
        inputs = self.dual_rnn(inputs)
        inputs = self.Prelu(inputs)
        inputs = self.conv2d(inputs)
        inputs = tf.split(inputs, self.num_spk, axis=-1)
        inputs = tf.concat(inputs, axis=0)
        inputs = over_add(inputs, gap)
        inputs = self.outputgate(inputs) * self.outputconv(inputs)
        inputs = self.end_conv1x1(inputs)
        inputs = tf.split(inputs, self.num_spk, axis=0)
        inputs = tf.cast(inputs, tf.float32)
        return inputs


class Model_basic(tf.keras.Model):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size=2,
                 dropout=0,
                 num_layers=4,
                 K=200,
                 num_spks=2):
        super(Model_basic, self).__init__()
        self.spk = num_spks
        self.enc = Encoder(kernel_size=kernel_size, out_channels=in_channels)
        self.dec = Decoder(kernel_size=kernel_size, num_speakers=1)
        self.sepration = DPRNNmodel(in_channels=in_channels,
                                    out_channels=out_channels,
                                    hidden_channels=hidden_channels,
                                    dropout=dropout,
                                    num_layers=num_layers,
                                    K=K,
                                    num_spks=num_spks)

    def call(self, inputs, training=None, mask=None):
        e = self.enc(inputs)
        s = self.sepration(e)
        separator = e[tf.newaxis] * s
        print(separator.shape)
        spk, bs, L, Chs = separator.get_shape().as_list()
        separator = tf.reshape(separator, [spk*bs, L, Chs])
        audios = self.dec(separator)
        audios = tf.reshape(audios, [spk, bs, -1])
        return audios


if __name__ == '__main__':
    tf.enable_eager_execution()
    test_inputs = np.random.normal(size=[4, 500, 1])
    test_inputs = tf.cast(test_inputs, tf.float32)
    Model_basic = Model_basic(in_channels=10, out_channels=10, hidden_channels=10, K=250)
    data = Model_basic(test_inputs)
    print(data.shape)
