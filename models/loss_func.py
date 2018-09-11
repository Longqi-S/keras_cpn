# --------------------------------------------------------------
# CPN: loss functions
# global loss + refine loss
# by Longqi-S
# --------------------------------------------------------------
import tensorflow as tf
import keras.backend as K
import keras.layers as KL

def global_net_loss(cfg, global_out, label, valids):
    """Loss for global net.
    global_outs: [batch, output_shape[0], output_shape[1], num_keypoints].
    labels     : [batch, output_shape[0], output_shape[1], num_keypoints].
    valids     : [batch, num_keypoints].
    """
    global_label = label * tf.to_float(tf.greater(tf.reshape(valids, (-1, 1, 1, cfg.KEYPOINTS_NUM)), 1.1))
    global_loss = tf.reduce_mean(tf.square(global_out - global_label))
    return global_loss

def global_net_losses(global_outs, labels, valids, cfg, name):
    """Loss for global net.
    global_outs: list of [batch, output_shape[0], output_shape[1], num_keypoints].
    labels     : list of [batch, output_shape[0], output_shape[1], num_keypoints].
    valids     : [batch, num_keypoints].
    list length: 4
    """
    global_losses = []
    for i, (global_out, label) in enumerate(zip(global_outs, labels)):
        global_loss = KL.Lambda(lambda x: global_net_loss(cfg, *x), name="global_net_loss{}".format(i + 1))(
                [global_out, label, valids])
        global_losses.append(global_loss)
    return KL.Lambda(lambda g: (g[0] + g[1] + g[2] + g[3]) / len(g) /2, name=name)(global_losses)


def ohkm(loss, top_k, cfg):
    ohkm_loss = 0.
    for i in range(cfg.IMAGES_PER_GPU):
        sub_loss = loss[i]
        topk_val, topk_idx = tf.nn.top_k(sub_loss, k=top_k, sorted=False, name='ohkm{}'.format(i))
        tmp_loss = tf.gather(sub_loss, topk_idx, name='ohkm_loss{}'.format(i)) # can be ignore ???
        ohkm_loss += tf.reduce_sum(tmp_loss) / top_k
    ohkm_loss /= cfg.IMAGES_PER_GPU
    return ohkm_loss

def refine_net_loss(refine_out, label, valids, cfg, name):
    def refine_net_loss_tf(cfg, refine_out, label, valids):
        refine_loss = tf.reduce_mean(tf.square(refine_out - label), (1,2)) * tf.to_float((tf.greater(valids, 0.1)))
        refine_loss = ohkm(refine_loss, 8, cfg)
        return refine_loss
    return KL.Lambda(lambda x: refine_net_loss_tf(cfg, *x), name=name)([refine_out, label, valids])