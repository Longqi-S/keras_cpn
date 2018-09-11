import keras.layers as KL
import keras.backend as K
import tensorflow as tf
from lib.nets.resnet_backbone import identity_block as bottleneck
from keras.utils import conv_utils
from keras.engine import InputSpec
import numpy as np

class UpsampleBilinear(KL.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = tf.shape(target)
        return tf.image.resize_bilinear(source, (target_shape[1], target_shape[2]), align_corners=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

def _conv_bn_relu(input_tensor, kernel_size, nb_filters,
                    padding="same", namebase="res", has_act=True, rate=1):
    output = KL.Conv2D(nb_filters, kernel_size, \
                padding=padding,
                dilation_rate=(rate, rate),
                name=namebase+"_conv")(input_tensor)
    output = KL.BatchNormalization(axis=3, \
                name=namebase+"_bn")(output)
    if has_act:
        output = KL.Activation('relu')(output)
    return output

def _bn_relu_conv(input_tensor, kernel_size, nb_filters,
                    padding="same", namebase="res", has_act=True):
    x = input_tensor
    x = KL.BatchNormalization(axis=3, \
                name=namebase+"_bn")(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(nb_filters, kernel_size, \
                padding=padding,
                name=namebase+"_conv")(x)
    return x
    
def create_global_net(blocks, cfg, has_bn=True, bn_trainable=True):
    """ create global net in cpn
    # Inputs:
    blocks = [C2, C3, C4, C5]
    """
    global_fms = []
    global_outs = []
    last_fm = None
    ## define pyramid feature maps
    for i, block in enumerate(reversed(blocks)):
        lateral = _conv_bn_relu(block, (1, 1), 256, "same", 'lateral/res{}'.format(5-i))
        if last_fm is not None:
            upsample = UpsampleBilinear(\
                name='fpn/p{}upsampled'.format(5-i+1))([last_fm, lateral])
            upsample = KL.Conv2D(256, (1, 1), \
                name='fpn/p{}upsampled_conv'.format(5-i))(upsample)
            if has_bn:
                upsample = KL.BatchNormalization(name='fpn/p{}upsampled_bn'.format(5-i), axis=3)(upsample)
            last_fm = KL.Add(name='fpn/p{}merge'.format(5-i))([\
                upsample, lateral])
        else:
            last_fm = lateral
        tmp = _conv_bn_relu(last_fm, (1, 1), 256, "SAME", 'tmp/res{}'.format(5-i))
        out = KL.Conv2D(cfg.KEYPOINTS_NUM, (3, 3), padding="SAME", \
                name='pyramid/res{}'.format(5-i))(tmp)
        if has_bn:
            out = KL.BatchNormalization(axis=3, name='pyramid/res{}_bn'.format(5-i))(out)
        global_fms.append(last_fm)
        out = KL.Lambda(lambda t: tf.image.resize_bilinear(t, \
                (cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1])), \
                name='pyramid/res{}up'.format(5-i))(out)
        global_outs.append(out)
    global_fms.reverse()
    global_outs.reverse()
    return global_fms, global_outs


## original cpn RefineNet version
def create_refine_net(blocks, cfg, use_bn=True):
    refine_fms = []
    for i, block in enumerate(blocks):
        mid_fm = block
        for j in range(i):
            mid_fm = bottleneck(mid_fm, 3, [128, 128, 256], 
                    stage=(2+i),
                    block='refine_conv' + str(j), use_bn=use_bn)
        mid_fm = KL.Lambda(lambda t: tf.image.resize_bilinear(t, \
                (cfg.OUTPUT_SHAPE[0], cfg.OUTPUT_SHAPE[1]), align_corners=True),\
                name='upsample_conv/res{}'.format(2+i))(mid_fm)
        refine_fms.append(mid_fm)
    refine_fm = KL.Concatenate(axis=3)(refine_fms)
    refine_fm = KL.Conv2D(256, (1, 1), 
                padding="SAME", name="refine_shotcut")(refine_fm)
    refine_fm = bottleneck(refine_fm, 3, [128, 128, 256], stage=0, block='final_bottleneck')
    res = KL.Conv2D(cfg.KEYPOINTS_NUM, (3, 3),
            padding='SAME', name='refine_out')(refine_fm)
    if use_bn:
        res = KL.BatchNormalization(name='refine_out_bn', axis=3)(res)
    return res