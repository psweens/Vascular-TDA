
import tensorflow as tf
import keras.backend as K
import keras
from keras.utils.conv_utils import convert_kernel
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

if K.backend() == 'theano':
    import theano
    import theano.tensor as T

def get_loss(loss):
    loss_dict = dict(adam='adam', sgd='sgd', self_weighted_binary_crossentropy=self_weighted_binary_crossentropy,
                     self_weighted_mean_squared_error=self_weighted_mean_squared_error,
                     binary_crossentropy='binary_crossentropy', mean_squared_error='mean_squared_error', mse='mean_squared_error',
                     gumbel_skeleton_loss_template=gumbel_skeleton_loss_template, gumbel_skeleton_loss_neighbour=gumbel_skeleton_loss_neighbour,
                     gumbel_sample_bce=gumbel_sample_bce, dice=dice,
                     skeleton_dice=skeleton_dice, bce='binary_crossentropy', weighted_bce=weighted_binary_crossentropy,
                     weighted_bce_2d=weighted_binary_crossentropy_2d, flat_bce=flat_binary_crossentropy)
    return loss_dict[loss]


def dice(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def skeleton_dice(y_true, y_pred):
    splits = theano.tensor.split(y_pred, [1,1,1,1,1], 5, axis=-3)
    sample = splits[1]

    splits_true = theano.tensor.split(y_true, [1,1,1,1,1], 5, axis=-3)
    sample_true = splits_true[1]

    return dice(sample_true, sample)

def weighted_binary_crossentropy_2d(y_true, y_pred):
    y_pred = K.batch_flatten(y_pred)

    kernel = K.ones((5, 5, 1, 1))
    kernel /= 25
    f = K.sum(y_true) + 1
    b = K.sum(1 - y_true) + 1

    t = f + b
    alpha = f / t

    tt2 = K.conv2d(y_true, kernel, padding='same')
    tt2 = K.conv2d(tt2, kernel, padding='same')
    tt2 = K.conv2d(tt2, kernel, padding='same')
    tt2 = K.conv2d(tt2, kernel, padding='same')
    tt2 = K.conv2d(tt2, kernel, padding='same')

    tt2 += alpha
    # tt2 += y_true * (1 - alpha) * K.max(tt2, keepdims=True)

    y_true = K.batch_flatten(y_true)
    y_true = K.clip(y_true, K.epsilon(), 1.0-K.epsilon())

    y_pred = tf.log(y_pred / (1 - y_pred))
    tt2 = K.batch_flatten(tt2)

    return K.mean(pixel_weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pixel_weight=tt2), axis=-1)

def flat_binary_crossentropy(y_true, y_pred):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    bce = K.binary_crossentropy(y_pred, y_true)
    return K.mean(bce)

def weighted_binary_crossentropy(y_true, y_pred):
    y_pred = K.batch_flatten(y_pred)
    kernel = K.ones((5,5,5,1,1))
    # kernel = K.variable(kernel)
    # tt = tf.Variable(initial_value=y_true, validate_shape=False)

    if K.image_data_format() == 'channels_first':
        tt = K.permute_dimensions(y_true, (0, 2, 1, 3, 4))
    else:
        tt = y_true + 0.
        # kernel = convert_kernel(kernel)

    kernel /= 125
    f = K.sum(y_true)
    b = K.sum(1 - y_true)

    t = f + b
    epsilon = 1
    alpha = K.sqrt((f+epsilon) / (t+epsilon))

    tt2 = K.conv3d(tt, kernel, padding='same', data_format=None)
    tt2 = K.conv3d(tt2, kernel, padding='same', data_format=None)
    tt2 = K.conv3d(tt2, kernel, padding='same', data_format=None)
    tt2 = K.conv3d(tt2, kernel, padding='same', data_format=None)
    tt2 = K.conv3d(tt2, kernel, padding='same', data_format=None)
    #
    if K.image_data_format() == 'channels_first':
        tt2 = K.permute_dimensions(tt2, (0, 2, 1, 3, 4))
    #
    tt2 += alpha
    tt2 += tt * (1 - alpha) * K.max(tt2, keepdims=True)
    #
    y_true = K.batch_flatten(y_true)
    #
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    #
    # cross_ent = -K.sum(y_true * K.log(y_pred) + (1-y_true)*K.log(1-y_pred))

    y_pred = tf.log(y_pred / (1 - y_pred))
    tt2 = K.batch_flatten(tt2)

    return K.mean(pixel_weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pixel_weight=tt2), axis=-1)
    # bce = (1.0/alpha) * y_true * K.log(y_pred) + (1.0/(1-alpha)) * (1.0-y_true) * K.log(1.0 - y_pred)
    # sample_weighted_bce = -K.flatten(tt2)*bce
    # sample_weighted_bce = K.binary_crossentropy(y_true, y_pred)

    # return K.sum(sample_weighted_bce) / K.sum(tt2)
    # return -K.sum(bce)


def junction_weighting(y_true, y_pred):
    import numpy as np
    y_pred = K.batch_flatten(y_pred)
    kernel = K.ones((5,5,5,1,1))

    neighbour_kernel = np.ones((3,3,3,1,1))
    neighbour_kernel[1,1,1,0,0] = 0
    neighbour_kernel = K.variable(neighbour_kernel)

    # kernel = K.variable(kernel)
    # tt = tf.Variable(initial_value=y_true, validate_shape=False)

    if K.image_data_format() == 'channels_first':
        tt = K.permute_dimensions(y_true, (0, 2, 1, 3, 4))
    else:
        tt = y_true + 0.
        # kernel = convert_kernel(kernel)

    kernel /= 125
    f = K.sum(y_true)
    b = K.sum(1 - y_true)

    t = f + b
    epsilon = 1
    alpha = K.sqrt((f+epsilon) / (t+epsilon))

    tt = K.conv3d(tt, neighbour_kernel, padding='same', data_format=None)
    tt2 = tt > 2

    tt2 = K.conv3d(tt2, kernel, padding='same', data_format=None)
    tt2 = K.conv3d(tt2, kernel, padding='same', data_format=None)
    tt2 = K.conv3d(tt2, kernel, padding='same', data_format=None)
    tt2 = K.conv3d(tt2, kernel, padding='same', data_format=None)
    tt2 = K.conv3d(tt2, kernel, padding='same', data_format=None)

    if K.image_data_format() == 'channels_first':
        tt2 = K.permute_dimensions(tt2, (0, 2, 1, 3, 4))
    #
    tt2 += alpha
    tt2 += tt * (1 - alpha) * K.max(tt2, keepdims=True)
    #
    y_true = K.batch_flatten(y_true)
    #
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    #
    # cross_ent = -K.sum(y_true * K.log(y_pred) + (1-y_true)*K.log(1-y_pred))

    y_pred = tf.log(y_pred / (1 - y_pred))
    tt2 = K.batch_flatten(tt2)

    return K.mean(pixel_weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pixel_weight=tt2), axis=-1)
    # bce = (1.0/alpha) * y_true * K.log(y_pred) + (1.0/(1-alpha)) * (1.0-y_true) * K.log(1.0 - y_pred)
    # sample_weighted_bce = -K.flatten(tt2)*bce
    # sample_weighted_bce = K.binary_crossentropy(y_true, y_pred)

    # return K.sum(sample_weighted_bce) / K.sum(tt2)
    # return -K.sum(bce)




def pixel_weighted_cross_entropy_with_logits(targets, logits, pixel_weight, name=None):
  """Adapts the tensorflow op weighted_cross_entropy_with_logits to give a vector of weights instead

  `logits` and `targets` must have the same type and shape.

  Args:
    targets: A `Tensor` of the same type and shape as `logits`.
    logits: A `Tensor` of type `float32` or `float64`.
    pixel_weight: A coefficient to use on the positive examples.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    weighted logistic losses.

  Raises:
    ValueError: If `logits` and `targets` do not have the same shape.
  """
  with ops.name_scope(name, "logistic_loss", [logits, targets]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    targets = ops.convert_to_tensor(targets, name="targets")
    try:
      targets.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and targets must have the same shape (%s vs %s)"
                       % (logits.get_shape(), targets.get_shape()))

    # The logistic loss formula from above is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
    # To avoid branching, we use the combined version
    #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
    log_weight = pixel_weight
    bce = math_ops.add((1 - targets) * logits,
        (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                      nn_ops.relu(-logits)),
        name=name)
    return math_ops.multiply(log_weight, bce)

def weighted_mean_squared_error(y_true, y_pred, weight_mask=None, delta=0.01):

    if weight_mask is not None:
        weight_mask = K.shared(weight_mask)
        weight_mask = K.flatten(weight_mask)
    else:
        weight_mask = y_true

    return K.mean((weight_mask + delta) * K.square(y_pred - y_true), axis=-1)


def unsupervised_intensity_loss(y_true, y_pred):

    return K.mean(K.flatten(y_pred))


def supervised_vness_loss(y_true, y_pred):


    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    return keras.objectives.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))

def self_weighted_binary_crossentropy(y_true, y_pred):

    epsilon = 0.1
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    y_mask = (y_true.copy() + epsilon)/(1+epsilon)

    return K.mean(y_mask*K.binary_crossentropy(y_pred, y_true))

def self_weighted_mean_squared_error(y_true, y_pred):

    epsilon = 0.1
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)

    y_mask = (y_true.copy() + epsilon)/(1+epsilon)

    return K.mean(y_mask*K.square(y_pred - y_true))


def unsupervised_vness_loss(y_true, y_pred):

    # return K.mean(K.flatten(K.pow(y_pred,2))) - K.mean(y_pred)
    # y_sq = K.flatten(K.pow(y_pred,2))
    # y_sum = K.sum(y_pred)

    # return -K.sum(y_sq)/K.pow(y_sum,2)

    return (K.sum(K.sigmoid(y_pred*100)) - K.log(K.mean(y_pred)))


def semisupervised_segmentation_loss(y_true, y_pred):
    """mean squared error loss allowing unlabeled and partially labelled images:)
        1 = foreground
        0 = background
        2 = unlabelled
    """

    mask = (y_true < 2).astype('uint8')

    nb_pixels = K.clip(K.sum(mask),1, 1e10)
    return K.sum((K.square(y_true - y_pred)*mask))/nb_pixels


def semisupervised_vness_loss(y_true, y_pred):
    """mean squared error loss allowing unlabeled and partially labelled images
        1 = foreground
        0 = background
        2 = unlabelled
    """
    mask_unlabelled = (y_true > 1).astype('uint8')
    nb_pixels = K.clip(K.sum(mask_unlabelled), 1, None)
    loss_unlabelled = -K.sum(y_pred * mask_unlabelled)/nb_pixels

    mask_labelled = (y_true < 2).astype('uint8')
    inv_true = (theano.tensor.ones(y_true.shape)-y_true)
    nb_foreground = K.clip(K.sum(y_pred*mask_labelled), 1, None)
    nb_background = K.clip(K.sum(inv_true*mask_labelled), 1, None)

    loss_labelled_pos = -K.sum(K.pow(y_pred*y_true*mask_labelled,2) * y_true)/nb_foreground

    loss_labelled_neg = K.sum(K.flatten(y_pred*inv_true*mask_labelled))/nb_background

    loss_labelled = loss_labelled_pos + loss_labelled_neg

    return loss_unlabelled + loss_labelled

def skeleton_loss(y_true, y_pred):

    splits = theano.tensor.split(y_pred, [1,1,1], 3, axis=-3)
    neighbours = splits[0]
    vness = splits[1]
    skeleton = splits[2]

    neighbours *= skeleton

    value_loss = 0.5*K.mean(K.log(K.flatten(skeleton)) + K.log(K.flatten(1-skeleton)))
    skel_loss = K.sum(K.square(neighbours-2)*y_true)/K.sum(y_true) + K.sum(K.square(neighbours)*(1-y_true))/K.sum((1-y_true))

    return skel_loss + value_loss

def gumbel_skeleton_loss_template(y_true, y_pred):

    from keras.layers import Lambda

    def softmax(E, axis=-1):
        e_x = K.exp(E)
        return e_x / K.sum(e_x,axis=axis, keepdims=True)

    # splits = K.split(y_pred, [1,2,2], 3, axis=-3)

    neighbours = Lambda(lambda x: x[:, :, :, 0])(y_pred)
    sample = Lambda(lambda x: x[:, :, :, 1:3])(y_pred)
    logits = Lambda(lambda x: x[:, :, :, 3:])(y_pred)

    sample = K.clip(sample, K.epsilon(), 1-K.epsilon())

    sample_f = Lambda(lambda x: x[:, :, :, 0])(sample)
    sample_b = Lambda(lambda x: x[:, :, :, 1])(sample)

    logits_f = Lambda(lambda x: x[:, :, :, 0])(logits)
    logits_b = Lambda(lambda x: x[:, :, :, 1])(logits)

    sigs_f = K.sigmoid(logits_f)

    neighbours_true = Lambda(lambda x: x[:, :, :, 0])(y_true)
    skeleton_true = Lambda(lambda x: x[:, :, :, 0])(y_true)

    q_y = logits
    q_y = softmax(q_y, axis=-1)
    log_q_y = K.log(q_y + K.epsilon())
    kl_tmp = q_y * (log_q_y - K.log(1.0/2.))
    KL = K.mean(kl_tmp)
    image_loss = keras.losses.binary_crossentropy(K.flatten(skeleton_true), K.flatten(sample_f))
    kl_loss = -KL
    skel_loss = K.sum(neighbours*sample_f)/(K.sum(skeleton_true)+K.epsilon())
    alpha = 0.5
    return alpha*image_loss + (1-alpha)*(skel_loss)

def gumbel_skeleton_loss_neighbour(y_true, y_pred):

    from keras.layers import Lambda

    def softmax(E, axis=-1):
        e_x = K.exp(E)
        return e_x / K.sum(e_x,axis=axis, keepdims=True)

    # splits = K.split(y_pred, [1,2,2], 3, axis=-3)

    neighbours = Lambda(lambda x: x[:, :, :, 0])(y_pred)
    sample = Lambda(lambda x: x[:, :, :, 1:3])(y_pred)
    logits = Lambda(lambda x: x[:, :, :, 3:])(y_pred)

    sample = K.clip(sample, K.epsilon(), 1-K.epsilon())

    sample_f = Lambda(lambda x: x[:, :, :, 0])(sample)
    sample_b = Lambda(lambda x: x[:, :, :, 1])(sample)

    logits_f = Lambda(lambda x: x[:, :, :, 0])(logits)
    logits_b = Lambda(lambda x: x[:, :, :, 1])(logits)

    sigs_f = K.sigmoid(logits_f)

    neighbours_true = Lambda(lambda x: x[:, :, :, 0])(y_true)
    skeleton_true = Lambda(lambda x: x[:, :, :, 0])(y_true)

    q_y = logits
    q_y = softmax(q_y, axis=-1)
    log_q_y = K.log(q_y + K.epsilon())
    kl_tmp = q_y * (log_q_y - K.log(1.0/2.))
    KL = K.mean(kl_tmp)
    image_loss = keras.losses.binary_crossentropy(K.flatten(skeleton_true), K.flatten(sample_f))
    kl_loss = -KL
    skel_loss = K.mean(K.square(neighbours-2)) + K.mean(K.square(neighbours))
    alpha = 0.5
    return alpha*image_loss + (1-alpha)*(skel_loss)

def gumbel_sample_bce(y_true, y_pred):

    splits = theano.tensor.split(y_pred, [1,2,2], 3, axis=-3)
    sample = splits[1]
    sample = K.clip(sample, K.epsilon(), 1-K.epsilon())
    splits_sample = theano.tensor.split(sample,[1,1],2,axis=-3)
    sample_f = splits_sample[0]

    splits_true = theano.tensor.split(y_true, [1,1,1,1,1], 5, axis=-3)
    skeleton_true = splits_true[3]

    gumbel_bce = keras.objectives.binary_crossentropy(K.flatten(skeleton_true), K.flatten(sample_f))

    return gumbel_bce

def skeleton_loss_templates(y_true, y_pred):

    splits = theano.tensor.split(y_pred, [1,1], 2, axis=-3)
    neighbours = splits[0]
    skeleton = splits[1]

    splits_true = theano.tensor.split(y_true, [1,1], 2, axis=-3)
    neighbours_true = splits[0]
    skeleton_true = splits[1]

    neighbours *= skeleton

    skeleton = K.clip(skeleton, K.epsilon(), 1-K.epsilon())

    value_loss = 0.5*K.mean(K.log(K.flatten(skeleton)) + K.log(K.flatten(1-skeleton)))
    skel_loss = K.sum(neighbours*skeleton_true)/K.sum(skeleton_true)
    cross_ent_loss = keras.objectives.binary_crossentropy(K.flatten(skeleton_true), K.flatten(skeleton))

    return 0.01*skel_loss + 0.0*value_loss + cross_ent_loss
