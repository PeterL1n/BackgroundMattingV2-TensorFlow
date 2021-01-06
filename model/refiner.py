import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU


class Refiner(Model):

    prevent_oversampling = True

    def __init__(self, mode, sample_pixels, threshold):
        super().__init__()
        assert mode in ['full', 'sampling', 'thresholding']

        self.mode = mode
        self.sample_pixels = sample_pixels
        self.threshold = threshold

        channels = [24, 16, 12, 4]
        self.conv1 = Conv2D(channels[0], 3, use_bias=False)
        self.bn1 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv2 = Conv2D(channels[1], 3, use_bias=False)
        self.bn2 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv3 = Conv2D(channels[2], 3, use_bias=False)
        self.bn3 = BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv4 = Conv2D(channels[3], 3, use_bias=True)
        self.relu = ReLU()

    def call(self, x, training=None):
        src, bgr, pha, fgr, err, hid = x
        
        H_full, W_full = tf.shape(src)[1], tf.shape(src)[2]
        H_half, W_half = H_full // 2, W_full // 2
        H_quat, W_quat = H_full // 4, W_full // 4

        src_bgr = tf.concat([src, bgr], axis=-1)
        
        if self.mode != 'full':
            err = tf.image.resize(err, (H_quat, W_quat))
            ref = self.select_refinement_regions(err)
            idx = tf.where(ref[:, :, :, 0])

            x = tf.concat([hid, pha, fgr], axis=-1)
            x = tf.image.resize(x, (H_half, W_half))
            x = self.crop_patch(x, idx, 2, 3)
            
            y = tf.image.resize(src_bgr, (H_half, W_half))
            y = self.crop_patch(y, idx, 2, 3)
            
            x = self.conv1(tf.concat([x, y], axis=-1), training=training)
            x = self.bn1(x, training=training)
            x = self.relu(x, training=training)
            x = self.conv2(x, training=training)
            x = self.bn2(x, training=training)
            x = self.relu(x, training=training)

            x = tf.image.resize(x, (8, 8), 'nearest')
            y = self.crop_patch(src_bgr, idx, 4, 2)
            
            x = self.conv3(tf.concat([x, y], axis=-1), training=training)
            x = self.bn3(x, training=training)
            x = self.relu(x, training=training)
            x = self.conv4(x, training=training)
            
            pha = tf.image.resize(pha, (H_full, W_full))
            pha = self.replace_patch(pha, x[:, :, :, :1], idx)

            fgr = tf.image.resize(fgr, (H_full, W_full))
            fgr = self.replace_patch(fgr, x[:, :, :, 1:], idx)
        else:
            x = tf.concat([hid, pha, fgr], axis=-1)
            x = tf.image.resize(x, (H_half, W_half))
            y = tf.image.resize(src_bgr, (H_half, W_half))

            x = tf.concat([x, y], axis=-1)
            x = tf.pad(x, tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]]))

            x = self.conv1(x, training=training)
            x = self.bn1(x, training=training)
            x = self.relu(x, training=training)
            x = self.conv2(x, training=training)
            x = self.bn2(x, training=training)
            x = self.relu(x, training=training)

            x = tf.image.resize(x, (H_full + 4, W_full + 4), 'nearest')
            y = tf.pad(src_bgr, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
            x = tf.concat([x, y], axis=-1)

            x = self.conv3(x, training=training)
            x = self.bn3(x, training=training)
            x = self.relu(x, training=training)
            x = self.conv4(x, training=training)

            pha = x[:, :, :, :1]
            fgr = x[:, :, :, 1:]
            ref = tf.ones((tf.shape(src)[0], 1, H_quat, W_quat), dtype=src.dtype)
        
        return pha, fgr, ref

    def crop_patch(self, x, idx, size: int, padding: int):
        box_indices = tf.cast(idx[:, 0], tf.dtypes.int32)
        
        y1 = idx[:, 1] * size - padding
        x1 = idx[:, 2] * size - padding
        y2 = idx[:, 1] * size + (size - 1) + padding
        x2 = idx[:, 2] * size + (size - 1) + padding
        
        shape = tf.shape(x)
        h = tf.cast(shape[1] - 1, tf.dtypes.float32)
        w = tf.cast(shape[2] - 1, tf.dtypes.float32)
        y1 = tf.cast(y1, tf.dtypes.float32) / h
        x1 = tf.cast(x1, tf.dtypes.float32) / w
        y2 = tf.cast(y2, tf.dtypes.float32) / h
        x2 = tf.cast(x2, tf.dtypes.float32) / w
        
        boxes = tf.stack([y1, x1, y2, x2], axis=1)
        return tf.image.crop_and_resize(x, boxes, box_indices, (size + 2 * padding, size + 2 * padding), 'nearest')
    
    def replace_patch(self, x, y, idx):
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1] // 4, 4, shape[2] // 4, 4, shape[3]))
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        x = tf.tensor_scatter_nd_update(x, idx, y)
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        x = tf.reshape(x, shape)
        return x
  
    def select_refinement_regions(self, err):
        if self.mode == 'sampling':
            k = self.sample_pixels // 16
            B = tf.shape(err)[0]
            H = tf.shape(err)[1]
            W = tf.shape(err)[2]
            idx = tf.reshape(err, (B, -1))
            idx = tf.math.top_k(idx, k=k, sorted=False)[1]
            idx = tf.stack([tf.tile(tf.range(B)[:, None], (1, k)), idx], axis=2)
            idx = tf.reshape(idx, (-1, 2))
            ref = tf.scatter_nd(idx, tf.ones(B * k), (B, H * W))
            ref = tf.reshape(ref, (B, H, W, 1))
            if self.prevent_oversampling:
                ref *= tf.cast(err > 0, err.dtype)
        else:
            ref = err > self.threshold
        return ref
