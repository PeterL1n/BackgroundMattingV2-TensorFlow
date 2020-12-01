import tensorflow as tf
from tensorflow.keras.models import Model

from .resnet import ResNetEncoder
from .mobilenet import MobileNetV2Encoder
from .aspp import ASPP
from .decoder import Decoder
from .refiner import Refiner


class MattingBase(Model):
    def __init__(self, backbone):
        super().__init__()
        assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetEncoder(backbone)
        else:
            self.backbone = MobileNetV2Encoder()
        self.aspp = ASPP(256, [3, 6, 9])
        self.decoder = Decoder([128, 64, 48, (1 + 3 + 1 + 32)])
    
    def call(self, x, training=None, _output_fgr_as_residual=False):
        src, bgr = x
        x = tf.concat([src, bgr], axis=-1)
        x, *shortcuts = self.backbone(x, training=training)
        x = self.aspp(x, training=training)
        x = self.decoder([x, *shortcuts], training=training)

        pha = tf.clip_by_value(x[:, :, :, 0:1], 0, 1)
        fgr = x[:, :, :, 1:4]
        err = tf.clip_by_value(x[:, :, :, 4:5], 0, 1)
        hid = tf.nn.relu(x[:, :, :, 5:])

        if not _output_fgr_as_residual:
            fgr = tf.clip_by_value(fgr + src, 0, 1)

        return pha, fgr, err, hid
        

class MattingRefine(MattingBase):
    def __init__(self,
                 backbone: str,
                 backbone_scale: float = 1/4,
                 refine_mode: str = 'sampling',
                 refine_sample_pixels: int = 80_000,
                 refine_threshold: float = 0.7):
        assert backbone_scale <= 1/2, 'backbone_scale should not be greater than 1/2'
        super().__init__(backbone)
        self.backbone_scale = backbone_scale
        self.refiner = Refiner(refine_mode, refine_sample_pixels, refine_threshold)

    def call(self, x, training=None):
        src, bgr = x
        tf.debugging.assert_equal(tf.shape(src), tf.shape(bgr),
                                  'src and bgr must have equal size.')
        tf.debugging.assert_equal(tf.shape(src)[1:3] // 4 * 4, tf.shape(src)[1:3],
                                  'src and bgr must have width and height that are divisible by 4')
        
        size_sm = tf.cast(tf.cast(tf.shape(src)[1:3], tf.float32) * self.backbone_scale, tf.int32)
        src_sm = tf.image.resize(src, size_sm)
        bgr_sm = tf.image.resize(bgr, size_sm)

        # Base
        pha_sm, fgr_sm, err_sm, hid_sm = super().call([src_sm, bgr_sm], training=training, _output_fgr_as_residual=True)

        # Refiner
        pha, fgr, ref_sm = self.refiner([src, bgr, pha_sm, fgr_sm, err_sm, hid_sm], training=training)

        # Clamp outputs
        pha = tf.clip_by_value(pha, 0, 1)
        fgr = tf.clip_by_value(fgr + src, 0, 1)
        fgr_sm = tf.clip_by_value(fgr_sm + src_sm, 0, 1)

        return pha, fgr, pha_sm, fgr_sm, err_sm, ref_sm
