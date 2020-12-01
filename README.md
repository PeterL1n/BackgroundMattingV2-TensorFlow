# Real-Time High-Resolution Background Matting (TensorFlow)

This repo contains TensorFlow 2 implementation of Real-Time High-Resolution Background Matting. For more information and downloading the weights, please visit our [official repo](https://github.com/PeterL1n/BackgroundMattingV2).

The TensorFlow implementation is experimental. We find PyTorch to have faster inference speed and suggest you to use the official PyTorch version whenever possible.

## Use our model

We reimplement our model natively in TensorFlow 2 and provide a script to load PyTorch weights directly into the TensorFlow model.

```python
import tensorflow as tf
import torch # For loading PyTorch weights.

from model import MattingRefine, load_torch_weights

# Create TensorFlow model
model = MattingRefine(backbone='resnet50',
                      backbone_scale=0.25,
                      refine_mode='sampling',
                      refine_sample_pixels=80000)

# Load PyTorch weights into TensorFlow model.
load_torch_weights(model, torch.load('PATH_TO_PYTORCH_WEIGHTS.pth'))

src = tf.random.normal((1, 1080, 1920, 3))
bgr = tf.random.normal((1, 1080, 1920, 3))

pha, fgr = model([src, bgr], training=False)[:2]
```

## Download weights

Please visit the [official repo](https://github.com/PeterL1n/BackgroundMattingV2) for detail.
