import tensorflow as tf
from tensorflow.python.keras import backend as k
tf.enable_eager_execution()
eps = 1e-6


class ConvBnRelu(tf.keras.Model):
    def __init__(self, filters, stride):
        super(ConvBnRelu, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=(1, stride), use_bias=False,
                                           padding="valid")
        self.bn = tf.keras.layers.BatchNormalization()
        self.pad = tf.keras.layers.ZeroPadding2D((1, 0))

    def call(self, inputs, training=None, mask=None):
        inputs = self.pad(inputs)
        inputs = self.conv(inputs)
        inputs = self.bn(inputs, training)
        inputs = tf.nn.elu(inputs)
        return inputs


class ConvBnReluinv(tf.keras.Model):
    def __init__(self, filters, stride):
        super(ConvBnReluinv, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=(1, stride), use_bias=False,
                                                    padding="valid")
        self.bn = tf.keras.layers.BatchNormalization()
        self.pad = tf.keras.layers.Cropping2D((1, 0))

    def call(self, inputs, training=None, mask=None):
        inputs = self.pad(inputs)
        inputs = self.conv(inputs)
        inputs = self.bn(inputs, training)
        inputs = tf.nn.elu(inputs)
        return inputs
    
    
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = ConvBnRelu(filters=16, stride=1)
        self.conv2 = ConvBnRelu(filters=16, stride=2)
        self.conv3 = ConvBnRelu(filters=32, stride=2)
        self.conv4 = ConvBnRelu(filters=64, stride=2)
        self.conv5 = ConvBnRelu(filters=128, stride=2)
        self.conv6 = ConvBnRelu(filters=256, stride=2)
        self.conv7 = ConvBnRelu(filters=512, stride=2)

        self.bilst1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=True),
                                                    merge_mode="concat")
        self.bilst2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=True),
                                                    merge_mode="concat")
        self.bilst3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=True),
                                                    merge_mode="concat")
        self.bilst4 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=512, return_sequences=True),
                                                    merge_mode="concat")

        self.convinv1 = ConvBnReluinv(filters=512, stride=2)
        self.convinv2 = ConvBnReluinv(filters=256, stride=2)
        self.convinv3 = ConvBnReluinv(filters=128, stride=2)
        self.convinv4 = ConvBnReluinv(filters=64, stride=2)
        self.convinv5 = ConvBnReluinv(filters=32, stride=2)
        self.convinv6 = ConvBnReluinv(filters=16, stride=2)
        self.convinv7 = ConvBnReluinv(filters=2, stride=1)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs, training)
        x2 = self.conv2(x1, training)
        x3 = self.conv3(x2, training)
        x4 = self.conv4(x3, training)
        x5 = self.conv5(x4, training)
        x6 = self.conv6(x5, training)
        x7 = self.conv7(x6, training)

        seqfeature = tf.squeeze(x7, axis=-2)
        seqfeature = self.bilst1(seqfeature)
        seqfeature = self.bilst2(seqfeature)
        seqfeature = self.bilst3(seqfeature)
        seqfeature = self.bilst4(seqfeature)

        seqfeature = tf.expand_dims(seqfeature, axis=-2)

        x7 = tf.concat([x7, seqfeature], axis=-1)
        x7 = self.convinv1(x7, training)

        x6 = tf.concat([x6, x7], axis=-1)
        x6 = self.convinv2(x6, training)

        x5 = tf.concat([x5, x6], axis=-1)
        x5 = self.convinv3(x5, training)

        x4 = tf.concat([x4, x5], axis=-1)
        x4 = self.convinv4(x4, training)

        x3 = tf.concat([x3, x4], axis=-1)
        x3 = self.convinv5(x3, training)

        x2 = tf.concat([x2, x3], axis=-1)
        x2 = self.convinv6(x2, training)

        x1 = tf.concat([x1, x2], axis=-1)
        x1 = self.convinv7(x1, training)

        return x1


if __name__ == '__main__':
    tf.enable_eager_execution()
    a = Generator()
    b = a(tf.ones([1, 128, 129, 1]), True)
    a.summary()
