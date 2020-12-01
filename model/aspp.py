import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dropout


class ASPP(Model):
    def __init__(self, filters, dilation_rates=[3, 6, 9]):
        super().__init__()
        self.aspp1 = ASPPConv(filters, 1, 1)
        self.aspp2 = ASPPConv(filters, 3, dilation_rates[0])
        self.aspp3 = ASPPConv(filters, 3, dilation_rates[1])
        self.aspp4 = ASPPConv(filters, 3, dilation_rates[2])
        self.pool = ASPPPooling(filters)
        self.project = Sequential([
            Conv2D(filters, 1, use_bias=False),
            BatchNormalization(momentum=0.1, epsilon=1e-5),
            ReLU(),
            Dropout(0.1)
        ])

    def call(self, x, training=None):
        x = tf.concat([
            self.aspp1(x, training=training),
            self.aspp2(x, training=training),
            self.aspp3(x, training=training),
            self.aspp4(x, training=training),
            self.pool(x, training=training)
        ], axis=-1)
        x = self.project(x, training=training)
        return x


class ASPPConv(Model):
    def __init__(self, filters, kernel_size, dilation_rate):
        super().__init__()
        self.conv = Conv2D(filters, kernel_size, padding='SAME', dilation_rate=dilation_rate, use_bias=False)
        self.bn = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.relu = ReLU()
    
    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.bn(x, training=training)
        x = self.relu(x, training=training)
        return x


class ASPPPooling(Model):
    def __init__(self, filters):
        super().__init__()
        self.pool = GlobalAveragePooling2D()
        self.conv = Conv2D(filters, 1, use_bias=False)
        self.bn = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.relu = ReLU()
    
    def call(self, x, training=None):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.pool(x, training=training)
        x = x[:, None, None, :]
        x = self.conv(x, training=training)
        x = self.bn(x, training=training)
        x = self.relu(x, training=training)
        x = tf.image.resize(x, (h, w), 'nearest')
        return x