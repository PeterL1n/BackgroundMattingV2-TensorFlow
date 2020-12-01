import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, ZeroPadding2D, DepthwiseConv2D


class MobileNetV2Encoder(Model):
    def __init__(self):
        super().__init__()
        self.features = [
            ConvBNReLU(32, 3, 2),
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6, 2),
            InvertedResidual(160, 160, 1, 6, 2),
            InvertedResidual(160, 320, 1, 6, 2),
        ]

    def call(self, x, training=None):
        x0 = x
        for i in range(0, 2):
            x = self.features[i](x, training=training)
        x1 = x
        for i in range(2, 4):
            x = self.features[i](x, training=training)
        x2 = x
        for i in range(4, 7):
            x = self.features[i](x, training=training)
        x3 = x
        for i in range(7, 18):
            x = self.features[i](x, training=training)
        x4 = x
        return x4, x3, x2, x1, x0


class InvertedResidual(Model):
    def __init__(self, inp, oup, strides, expand_ratio, dilation_rate=1):
        super().__init__()
        self.use_res_connect = strides == 1 and inp == oup    
        hidden_filters = int(round(inp * expand_ratio))
        
        if expand_ratio != 1:
            self.pw = ConvBNReLU(hidden_filters, 1)
        self.dw = ConvBNReLU(hidden_filters, 3, strides, True, dilation_rate)
        self.pw_linear = ConvBNReLU(oup, 1, activation=False)
    
    def call(self, x, training=None):
        identity = x
        if hasattr(self, 'pw'):
            x = self.pw(x, training=training)
        x = self.dw(x, training=training)
        x = self.pw_linear(x, training=training)
        if self.use_res_connect:
            x += identity
        return x


class ConvBNReLU(Sequential):
    def __init__(self, filters, kernel_size, strides=1, depthwise=False, dilation_rate=1, activation=True):
        super().__init__()
        padding = (kernel_size * dilation_rate - 1) // 2
        if padding != 0:
            self.pad = ZeroPadding2D((padding, padding))
        if depthwise:
            self.conv = DepthwiseConv2D(kernel_size, strides, dilation_rate=dilation_rate, use_bias=False)
        else:
            self.conv = Conv2D(filters, kernel_size, strides, dilation_rate=dilation_rate, use_bias=False)
        self.bn = BatchNormalization(momentum=0.1, epsilon=1e-5)
        if activation:
            self.relu = ReLU(6)

    def call(self, x, training=None):
        if hasattr(self, 'pad'):
            x = self.pad(x, training=training)
        x = self.conv(x, training=training)
        x = self.bn(x, training=training)
        if hasattr(self, 'relu'):
            x = self.relu(x, training=training)
        return x