import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU


class Decoder(Model):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv2D(channels[0], 3, padding='SAME', use_bias=False)
        self.bn1 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv2 = Conv2D(channels[1], 3, padding='SAME', use_bias=False)
        self.bn2 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv3 = Conv2D(channels[2], 3, padding='SAME', use_bias=False)
        self.bn3 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv4 = Conv2D(channels[3], 3, padding='SAME', use_bias=True)
        self.relu = ReLU()

    def call(self, x, training=None):
        x4, x3, x2, x1, x0 = x
        x = tf.image.resize(x4, tf.shape(x3)[1:3])
        x = tf.concat([x, x3], axis=-1)
        x = self.conv1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.relu(x, training=training)
        x = tf.image.resize(x, tf.shape(x2)[1:3])
        x = tf.concat([x, x2], axis=-1)
        x = self.conv2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.relu(x, training=training)
        x = tf.image.resize(x, tf.shape(x1)[1:3])
        x = tf.concat([x, x1], axis=-1)
        x = self.conv3(x, training=training)
        x = self.bn3(x, training=training)
        x = self.relu(x, training=training)
        x = tf.image.resize(x, tf.shape(x0)[1:3])
        x = tf.concat([x, x0], axis=-1)
        x = self.conv4(x, training=training)
        return x