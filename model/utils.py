import tensorflow as tf

from .resnet import ResNetEncoder
from .mobilenet import MobileNetV2Encoder


def load_torch_weights(model, state_dict, default_size=(1080, 1920)):
    # Build model
    model([tf.random.normal((1, *default_size, 3)), tf.random.normal((1, *default_size, 3))], training=False)
        
    # ResNet backbone
    if isinstance(model.backbone, ResNetEncoder):
        load_conv_weights(model.backbone.conv1, state_dict, 'backbone.conv1')
        load_bn_weights(model.backbone.bn1, state_dict, 'backbone.bn1')
        for l in range(1, 5):
            for b, resblock in enumerate(getattr(model.backbone, f'layer{l}').layers):
                if hasattr(resblock, 'convd'):
                    load_conv_weights(resblock.convd, state_dict, f'backbone.layer{l}.{b}.downsample.0')
                    load_bn_weights(resblock.bnd, state_dict, f'backbone.layer{l}.{b}.downsample.1')
                load_conv_weights(resblock.conv1, state_dict, f'backbone.layer{l}.{b}.conv1')
                load_conv_weights(resblock.conv2, state_dict, f'backbone.layer{l}.{b}.conv2')
                load_conv_weights(resblock.conv3, state_dict, f'backbone.layer{l}.{b}.conv3')
                load_bn_weights(resblock.bn1, state_dict, f'backbone.layer{l}.{b}.bn1')
                load_bn_weights(resblock.bn2, state_dict, f'backbone.layer{l}.{b}.bn2')
                load_bn_weights(resblock.bn3, state_dict, f'backbone.layer{l}.{b}.bn3')

    # MobileNet backbone
    if isinstance(model.backbone, MobileNetV2Encoder):
        load_conv_weights(model.backbone.features[0].conv, state_dict, 'backbone.features.0.0')
        load_bn_weights(model.backbone.features[0].bn, state_dict, 'backbone.features.0.1')
        load_conv_weights(model.backbone.features[1].dw.conv, state_dict, 'backbone.features.1.conv.0.0', True)
        load_bn_weights(model.backbone.features[1].dw.bn, state_dict, 'backbone.features.1.conv.0.1')
        load_conv_weights(model.backbone.features[1].pw_linear.conv, state_dict, 'backbone.features.1.conv.1')
        load_bn_weights(model.backbone.features[1].pw_linear.bn, state_dict, 'backbone.features.1.conv.2')
        for i in range(2, 18):
            load_conv_weights(model.backbone.features[i].pw.conv, state_dict, f'backbone.features.{i}.conv.0.0')
            load_bn_weights(model.backbone.features[i].pw.bn, state_dict, f'backbone.features.{i}.conv.0.1')
            load_conv_weights(model.backbone.features[i].dw.conv, state_dict, f'backbone.features.{i}.conv.1.0', True)
            load_bn_weights(model.backbone.features[i].dw.bn, state_dict, f'backbone.features.{i}.conv.1.1')
            load_conv_weights(model.backbone.features[i].pw_linear.conv, state_dict, f'backbone.features.{i}.conv.2')
            load_bn_weights(model.backbone.features[i].pw_linear.bn, state_dict, f'backbone.features.{i}.conv.3')

    # ASPP
    for i in range(4):
        load_conv_weights(getattr(model.aspp, f'aspp{i+1}').conv, state_dict, f'aspp.convs.{i}.0')
        load_bn_weights(getattr(model.aspp, f'aspp{i+1}').bn, state_dict, f'aspp.convs.{i}.1')
    load_conv_weights(model.aspp.pool.conv, state_dict, f'aspp.convs.4.1')
    load_bn_weights(model.aspp.pool.bn, state_dict, f'aspp.convs.4.2')
    load_conv_weights(model.aspp.project.layers[0], state_dict, f'aspp.project.0')
    load_bn_weights(model.aspp.project.layers[1], state_dict, f'aspp.project.1')

    # Decoder
    load_conv_weights(model.decoder.conv1, state_dict, 'decoder.conv1')
    load_bn_weights(model.decoder.bn1, state_dict, 'decoder.bn1')
    load_conv_weights(model.decoder.conv2, state_dict, 'decoder.conv2')
    load_bn_weights(model.decoder.bn2, state_dict, 'decoder.bn2')
    load_conv_weights(model.decoder.conv3, state_dict, 'decoder.conv3')
    load_bn_weights(model.decoder.bn3, state_dict, 'decoder.bn3')
    load_conv_weights(model.decoder.conv4, state_dict, 'decoder.conv4')
    
    # Refiner
    if hasattr(model, 'refiner'):
        load_conv_weights(model.refiner.conv1, state_dict, 'refiner.conv1')
        load_bn_weights(model.refiner.bn1, state_dict, 'refiner.bn1')
        load_conv_weights(model.refiner.conv2, state_dict, 'refiner.conv2')
        load_bn_weights(model.refiner.bn2, state_dict, 'refiner.bn2')
        load_conv_weights(model.refiner.conv3, state_dict, 'refiner.conv3')
        load_bn_weights(model.refiner.bn3, state_dict, 'refiner.bn3')
        load_conv_weights(model.refiner.conv4, state_dict, 'refiner.conv4')

def load_conv_weights(conv, state_dict, name, depthwise_conv=False):
    weight = state_dict[name + '.weight']
    if depthwise_conv:
        weight = weight.permute(2, 3, 0, 1).numpy()
    else:
        weight = weight.permute(2, 3, 1, 0).numpy()
    if name + '.bias' in state_dict:
        bias = state_dict[name + '.bias'].numpy()
        conv.set_weights([weight, bias])
    else:
        conv.set_weights([weight])

def load_bn_weights(bn, state_dict, name):
    weight = state_dict[name + '.weight']
    bias = state_dict[name + '.bias']
    running_mean = state_dict[name + '.running_mean']
    running_var = state_dict[name + '.running_var']
    bn.set_weights([weight, bias, running_mean, running_var])