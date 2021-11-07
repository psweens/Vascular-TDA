from keras.models import Model
from keras.layers import Input, UpSampling2D, Reshape
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate, Concatenate
import numpy as np
from unet_core.networks.layers import ReflectPadding2D, KernelFilter, VesselnessLayer, GumbelSkeletonLayer
from unet_core.networks.loss_functions import get_loss

from keras.layers import merge, Dropout, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D, Convolution3D, MaxPooling3D, UpSampling3D, Conv2D, Conv3D
from keras.layers.normalization import BatchNormalization

from keras import backend as K


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def _get_network(params):

    architecture = params["architecture"]
    initial_weights = params["initial_weights"]

    if architecture == 'unet1':
        model = unet1(**params)
    elif architecture == 'unet2':
        model = unet2(**params)
    elif architecture =='mini_net':
        model = mini_net(**params)
    elif architecture =='micro_net':
        model = micro_net(**params)
    elif architecture == 'inceptionv4_U':
        model = inceptionv4_U(**params)
    elif architecture == 'conv_lstm':
        model = conv_lstm(**params)
    else:
        raise ValueError('Invalid architecture')

    return model


def vessel_unit(input_layer, nb_scale_layers=0, nb_filters=1, alpha=0.5, beta=0.5, gamma=10):

    scale_layer = input_layer

    kernel_size = 3
    ks = kernel_size

    smooth_kernel = build_smooth_kernel((1, 1, ks, ks))

    scale_layer = KernelFilter(smooth_kernel, border_mode='valid')(scale_layer)
    scale_layer = ReflectPadding2D(padding=1, batch_ndim=2)(scale_layer)
    v_layer = VesselnessLayer(alpha=alpha, beta=beta, gamma=gamma)(scale_layer)
    # v_layer = KernelFilter(smooth_kernel, border_mode='same')(v_layer)

    if nb_scale_layers > 0:
        scale_layers = []
        scale_layers.append(v_layer)

        for l in range(nb_scale_layers):
            scale_layer = KernelFilter(smooth_kernel, border_mode='valid')(scale_layer)
            scale_layer = ReflectPadding2D(padding=1, batch_ndim=2)(scale_layer)
            v_layer = VesselnessLayer(alpha=alpha, beta=beta, gamma=gamma)(scale_layer)
            v_layer = KernelFilter(smooth_kernel, border_mode='valid')(v_layer)
            v_layer = ReflectPadding2D(padding=1, batch_ndim=2)(v_layer)
            scale_layers.append(v_layer)

        output_layer = merge(scale_layers, mode='max', concat_axis=-1)
    else:
        output_layer = v_layer

    return output_layer


def build_smooth_kernel(shape):
    ks = shape[-1]
    c = (ks-1)/2

    sigma = np.sqrt(1.)
    x, y = np.mgrid[0:ks, 0:ks]
    x = x.astype(float)
    y = y.astype(float)

    kernel = np.exp(-((x-c)**2 + (y-c)**2)/sigma**2)
    kernel /= np.sum(kernel)

    """Scale Space Correction Factor (http://www.cim.mcgill.ca/~langer/558/2009/lecture11.pdf)"""
    kernel *= sigma

    return kernel.reshape(shape)


def micro_net(params):
    kernel_size = (params["conv_size"], params["conv_size"])
    img_rows, img_cols = params["patch_size"][0], params["patch_size"][1]
    nb_pool = 2
    p = 0.5

    init_style = params["init_style"]
    input_shape = (params["num_channels"], img_rows, img_cols)

    inputs = Input(input_shape)
    conv1 = Convolution2D(2, kernel_size[0], kernel_size[1], init=init_style, border_mode='same', activation='relu', input_shape=input_shape)(inputs)
    conv1 = Convolution2D(2, kernel_size[0], kernel_size[1], init=init_style, border_mode='same', activation='relu')(conv1)
    conv1 = Convolution2D(2, kernel_size[0], kernel_size[1], init=init_style, border_mode='same', activation='relu')(conv1)

    conv1 = Convolution2D(1, 1, 1, border_mode='same')(conv1)

    flat1 = Flatten()(conv1)

    if params['training_phase'] == 'supervised':
        act1 = Activation('sigmoid')(flat1)
        reshape1 = Reshape((1, img_rows, img_cols))(act1)
        model = Model(input=inputs, output=reshape1)
        model.compile(loss=dice_coef_loss, metrics=[dice_coef], optimizer='adam')
    elif params['training_phase'] == 'pre_train':
        reshape1 = Activation('softmax')(flat1)
        # reshape1 = Reshape((1, img_rows, img_cols))(reshape1)
        model = Model(input=inputs, output=reshape1)
        model.compile(loss='binary_crossentropy', optimizer='adam')
    else:
        raise ValueError('Invalid training_phase')

    return model


def mini_net(tile_size=None, channels=[0,], loss='mean_squared_error', init_style='glorot_uniform',
             final_activation='sigmoid', optimizer='adam', **kwargs):

    nb_filters = [16, 32, 64]
    kernel_size = (3, 3)
    img_rows, img_cols = tile_size[0], tile_size[1]
    nb_pool = 2
    num_channels=len(channels)

    input_shape = (num_channels, img_rows, img_cols)
    inputs = Input(input_shape)
    conv1 = Convolution2D(nb_filters[1], kernel_size[0], kernel_size[1], init=init_style, border_mode='same', activation='relu', input_shape=input_shape)(inputs)
    pool1 = MaxPooling2D(pool_size=(nb_pool, nb_pool))(conv1)
    conv3 = Convolution2D(nb_filters[2], kernel_size[0], kernel_size[1], init=init_style, border_mode='same', activation='relu')(pool1)
    up2 = merge([UpSampling2D(size=(2, 2))(conv3), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(nb_filters[1], kernel_size[0], kernel_size[1], init=init_style, border_mode='same', activation='relu')(up2)
    conv6 = Convolution2D(1, 1, 1, init=init_style, border_mode='same')(conv5)

    act1 = Activation(final_activation)(conv6)

    model = Model(input=inputs, output=act1)
    model.compile(loss=get_loss(loss), optimizer=optimizer)

    return model

def mini_net_skeleton(tau=2.0, tile_size=None, channels=[0,], loss='mean_squared_error', init_style='glorot_uniform',
             final_activation='sigmoid', learning_rate=0.001, optimizer='adam', **kwargs):

    nb_filters = [16, 32, 64]
    kernel_size = (3, 3)
    img_rows, img_cols = tile_size[0], tile_size[1]
    nb_pool = 2
    num_channels=len(channels)

    input_shape = (num_channels, img_rows, img_cols)
    inputs = Input(input_shape)
    conv1 = Convolution2D(nb_filters[1], kernel_size[0], kernel_size[1], init=init_style, border_mode='same', activation='relu', input_shape=input_shape)(inputs)
    pool1 = MaxPooling2D(pool_size=(nb_pool, nb_pool))(conv1)
    conv3 = Convolution2D(nb_filters[2], kernel_size[0], kernel_size[1], init=init_style, border_mode='same', activation='relu')(pool1)
    up2 = merge([UpSampling2D(size=(2, 2))(conv3), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(nb_filters[1], kernel_size[0], kernel_size[1], init=init_style, border_mode='same', activation='relu')(up2)
    conv6 = Convolution2D(1, 1, 1, init=init_style, border_mode='same')(conv5)

    act1 = Activation(final_activation)(conv6)
    K_tau = K.variable(tau)
    skel1 = GumbelSkeletonLayer(K_tau)(act1)
    model = Model(input=inputs, output=skel1)
    model.compile(loss=get_loss(loss), optimizer='adam')
    model.tau = K_tau

    return model


def unet1(**kwargs):
    return unet(convs_per_layer=1, is_3d=False, **kwargs)


def unet2(**kwargs):
    kwargs['is_3d'] = False
    return unet(convs_per_layer=2, **kwargs)


def unet3(**kwargs):
    kwargs['is_3d'] = False
    return unet(convs_per_layer=3, is_3d=False, **kwargs)


def unet4(**kwargs):
    kwargs['is_3d'] = False
    return unet(convs_per_layer=4, is_3d=False, **kwargs)

def unet_3d(**kwargs):
    kwargs['is_3d'] = True
    return unet(**kwargs)


def unet2_skeleton(tau=100.0, mode='template', learning_rate=1e-4, **kwargs):

    if mode=='template':
        loss = 'gumbel_skeleton_loss_template'
    if mode == 'neighbour':
        loss = 'gumbel_skeleton_loss_neighbour'

    K_tau = K.variable(tau)
    input, output = unet2(build_model=False, **kwargs)
    skel1 = GumbelSkeletonLayer(K_tau, mode=mode)(output)
    model = Model(input=input, output=skel1)
    model.compile(loss=get_loss(loss), optimizer=Adam(lr=learning_rate))
    model.tau = K_tau
    return model


def unet(is_3d=False, convs_per_layer=2, tile_size=None, channels=None, loss='mean_squared_error', init_style='glorot_uniform',
         internal_dropout_rate=0, final_dropout_rate=0.5, final_activation='sigmoid', flat_output=False,
         learning_rate=1e-4, batch_norm=False, compile=True, input_layer=None, reporter_loss=None, filters=None,
         build_model=True, **kwargs):

    if tile_size is None:
        tile_size = [512, 512]
    elif not is_3d:
        tile_size = tile_size[:2]

    if channels is None:
        channels = [0, ]

    num_channels = len(channels)

    if K.image_data_format() == 'channels_first':
        input_shape = (num_channels,) + tuple(tile_size)
        concat_axis = -3
    elif K.image_data_format() == 'channels_last':
        input_shape = tuple(tile_size) + (num_channels,)
        concat_axis = -1

    if input_layer is not None:
        inputs = input_layer
    else:
        inputs = Input(input_shape)

    if filters is None:
        filters = [32, 64, 128, 128, 128, 128, 128, 64, 32]

    if is_3d:
        pool_size = (2,2,1)
        down_pool = MaxPooling3D
        up_pool = UpSampling3D
    else:
        pool_size = (2,2)
        down_pool = MaxPooling2D
        up_pool = UpSampling2D

    conv1 = unet_unit(inputs, nb_filters=filters[0], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate, is_3d=is_3d)
    pool1 = down_pool(pool_size=pool_size)(conv1)

    conv2 = unet_unit(pool1, nb_filters=filters[1], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate, is_3d=is_3d)
    pool2 = down_pool(pool_size=pool_size)(conv2)

    conv3 = unet_unit(pool2, nb_filters=filters[2], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate, is_3d=is_3d)
    pool3 = down_pool(pool_size=pool_size)(conv3)

    conv4 = unet_unit(pool3, nb_filters=filters[3], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate, is_3d=is_3d)
    pool4 = down_pool(pool_size=pool_size)(conv4)

    conv5 = unet_unit(pool4, nb_filters=filters[4], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate, is_3d=is_3d)

    up6 = concatenate([up_pool(size=pool_size)(conv5), conv4], axis=concat_axis)
    conv6 = unet_unit(up6, nb_filters=filters[-4], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate, is_3d=is_3d)

    up7 = concatenate([up_pool(size=pool_size)(conv6), conv3], axis=concat_axis)
    conv7 = unet_unit(up7, nb_filters=filters[-3], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate, is_3d=is_3d)

    up8 = concatenate([up_pool(size=pool_size)(conv7), conv2], axis=concat_axis)
    conv8 = unet_unit(up8, nb_filters=filters[-2], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate, is_3d=is_3d)

    up9 = concatenate([up_pool(size=pool_size)(conv8), conv1], axis=concat_axis)
    conv9 = unet_unit(up9, nb_filters=filters[-1], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate, is_3d=is_3d)

    conv9 = Dropout(final_dropout_rate)(conv9)
    if is_3d:
        conv10 = Conv3D(1, (1, 1, 1))(conv9)
    else:
        conv10 = Conv2D(1, (1, 1))(conv9)

    act1 = Activation(final_activation, name='regular')(conv10)
    reporter = Activation(final_activation, name='reporter')(conv10)

    if not build_model:
        return inputs, act1
    else:
        if compile:
            if reporter_loss is not None:
                model = Model(input=inputs, output=[act1, reporter])
                model.compile(loss=[get_loss(loss), get_loss(reporter_loss)],
                              optimizer=Adam(lr=learning_rate),
                              loss_weights=[1, 0])
            else:
                model = Model(input=inputs, output=act1)
                model.compile(loss=get_loss(loss), optimizer=Adam(lr=learning_rate))
        else:
            if reporter_loss is not None:
                model = Model(input=inputs, output=[act1, reporter])
            else:
                model = Model(input=inputs, output=act1)

        return model


def unet_sub_unit(in_layer, nb_filters=32, init_style='glorot_uniform', dropout_rate=0.1, batch_norm=False, leaky_activation=False,
             filter_size=3, is_3d=False, **kwargs):
    cs = filter_size

    if is_3d:
        c1 = Conv3D(nb_filters, (cs, cs, cs), kernel_initializer=init_style, padding='same')(in_layer)
    else:
        c1 = Conv2D(nb_filters, (cs, cs), kernel_initializer=init_style, padding='same')(in_layer)

    if batch_norm:
        if K.image_data_format() == 'channels_first':
            c1 = BatchNormalization(axis=1)(c1)
        elif K.image_data_format() == 'channels_last':
            c1 = BatchNormalization(axis=-1)(c1)

    if leaky_activation:
        c1 = LeakyReLU(0.1)(c1)
    else:
        c1 = Activation('relu')(c1)

    if dropout_rate > 0.0:
        c1 = Dropout(dropout_rate)(c1)

    return c1

def unet_unit(input, nb_layers=2,  **kwargs):

    conv1 = unet_sub_unit(input, **kwargs)
    for i in range(nb_layers-1):
        conv1 = unet_sub_unit(conv1, **kwargs)

    return conv1


"""
Implementation of Inception Network v4 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.
"""


if K.image_dim_ordering() == "th":
    channel_axis = 1
else:
    channel_axis = -1


def inception_stem(input): # Input (299,299,3)
    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2,2))(input)
    c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)

    c1 = MaxPooling2D((3,3), strides=(2,2))(c)
    c2 = Convolution2D(96, 3, 3, activation='relu', subsample=(2,2))(c)

    m = merge([c1, c2], mode='concat', concat_axis=channel_axis)

    c1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c1 = Convolution2D(96, 3, 3, activation='relu', )(c1)

    c2 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c2 = Convolution2D(64, 7, 1, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(64, 1, 7, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid')(c2)

    m2 = merge([c1, c2], mode='concat', concat_axis=channel_axis)

    p1 = MaxPooling2D((3,3), strides=(2,2), )(m2)
    p2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2,2))(m2)

    m3 = merge([p1, p2], mode='concat', concat_axis=channel_axis)
    m3 = BatchNormalization(axis=1)(m3)
    m3 = Activation('relu')(m3)
    return m3

def inception_mini(input, nb_filters):
    nb_filters_1 = nb_filters
    nb_filters_2 = int(1.5*nb_filters)

    a1 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    c1 = Convolution2D(nb_filters_1, 1, 1, activation='relu', border_mode='same')(a1)

    c2 = Convolution2D(nb_filters_1, 1, 1, activation='relu', border_mode='same')(input)

    c3 = Convolution2D(nb_filters_2, 1, 1, activation='relu', border_mode='same')(input)
    c3_1 = Convolution2D(nb_filters_1, 1, 3, activation='relu', border_mode='same')(c3)
    c3_2 = Convolution2D(nb_filters_1, 3, 1, activation='relu', border_mode='same')(c3)

    m = merge([c1, c2, c3_1, c3_2], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m


def inception_A(input, nb_filters=96):
    a1 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    a1 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(a1)

    a2 = Convolution2D(96, 1, 1, activation='relu', border_mode='same')(input)

    a3 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    a3 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a3)

    a4 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(input)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a4)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same')(a4)

    m = merge([a1, a2, a3, a4], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m

def inception_B(input, nb_filters=384):

    filters = [nb_filters]

    b1 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    b1 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(b1)

    b2 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)

    b3 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    b3 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(b3)
    b3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(b3)

    b4 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    b4 = Convolution2D(192, 1, 7, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(224, 7, 1, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(224, 1, 7, activation='relu', border_mode='same')(b4)
    b4 = Convolution2D(256, 7, 1, activation='relu', border_mode='same')(b4)

    m = merge([b1, b2, b3, b4], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m

def inception_C(input, nb_filters=256):
    c1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c1 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(c1)

    c2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)

    c3 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    c3_1 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(c3)
    c3_2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(c3)

    c4 = Convolution2D(384, 1, 1, activation='relu', border_mode='same')(input)
    c4 = Convolution2D(192, 1, 3, activation='relu', border_mode='same')(c4)
    c4 = Convolution2D(224, 3, 1, activation='relu', border_mode='same')(c4)
    c4_1 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(c4)
    c4_2 = Convolution2D(256, 1, 3, activation='relu', border_mode='same')(c4)

    m = merge([c1, c2, c3_1, c3_2, c4_1, c4_2], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m

def reduction_A(input, k=192, l=224, m=256, n=384):
    r1 = MaxPooling2D((3,3), strides=(2,2))(input)

    r2 = Convolution2D(n, 3, 3, activation='relu', subsample=(2,2))(input)

    r3 = Convolution2D(k, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(l, 3, 3, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(m, 3, 3, activation='relu', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m

def reduction_B(input):
    r1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(input)

    r2 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    r2 = Convolution2D(192, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(320, 7, 1, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(320, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m


def inceptionv4_U(params):
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)

    init_style = params["init_style"]
    cs = params["conv_size"]

    # nb_filters = 32
    nb_filters = [32, 64, 128]
    kernel_size = (cs, cs)
    img_rows, img_cols = params["patch_size"][0], params["patch_size"][1]
    nb_pool = 2
    p = 0.5
    final_activation = params['final_activation']
    flat_output = params['flat_output']
    learning_rate = params['learning_rate']
    optimizer = params['optimizer']
    loss = params['loss']

    input_shape = (params["num_channels"], img_rows, img_cols)

    inputs = Input(input_shape)

    #   Inception stem
    c = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    c = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(c)
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)

    c1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(c)
    c2 = Convolution2D(96, 3, 3, activation='relu', subsample=(2, 2), border_mode='same')(c)

    m1 = merge([c1, c2], mode='concat', concat_axis=channel_axis)

    m2 = inception_mini(m1, nb_filters=64)

    p1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(m2)

    m3 = inception_mini(p1, nb_filters=128)

    p3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(m3)

    m4 = inception_mini(p3, nb_filters=256)

    p4 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(m4)

    I1 = inception_mini(p4, nb_filters=512)

    up1 = merge([UpSampling2D(size=(2, 2))(I1), m4], mode='concat', concat_axis=1)

    m5 = inception_mini(up1, nb_filters=256)

    up2 = merge([UpSampling2D(size=(2, 2))(m4), m5], mode='concat', concat_axis=1)

    m6 = inception_mini(up2, nb_filters=128)

    up3 = merge([UpSampling2D(size=(2, 2))(m6), m2], mode='concat', concat_axis=1)

    m7 = inception_mini(up3, nb_filters=64)

    up4 = merge([UpSampling2D(size=(2, 2))(m7), c], mode='concat', concat_axis=1)

    m8 = inception_mini(up4, nb_filters=32)
    m8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(m8)

    conv10 = Convolution2D(1, 1, 1)(m8)
    conv10 = Dropout(0.5)(conv10)
    flat1 = Flatten()(conv10)

    act1 = Activation(final_activation)(flat1)
    if flat_output is False:
        act1 = Reshape((1, img_rows, img_cols))(act1)

    model = Model(input=inputs, output=act1)
    model.compile(loss=loss, optimizer=Adam(lr=learning_rate))

    return model


    # x = inception_stem(input)
    #
    # # 4 x Inception A
    # x = inception_A(x)
    # x = inception_A(x)
    # x = inception_A(x)
    # x = inception_A(x)
    #
    # # Reduction A
    # x = reduction_A(x)
    #
    # # 7 x Inception B
    # x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)
    # x = inception_B(x)
    #
    # # Reduction B
    # x = reduction_B(x)
    #
    # # 3 x Inception C
    # x = inception_C(x)
    # x = inception_C(x)
    # x = inception_C(x)
    #
    # # Average Pooling
    # x = AveragePooling2D((8,8))(x)
    #
    # # Dropout
    # x = Dropout(0.8)(x)
    # x = Flatten()(x)
    #
    # # Output
    # x = Dense(output_dim=nb_output, activation='softmax')(x)
    # return x

def conv_lstm(params):
    from conv_gru import LSTMConv2D
    from keras.layers.recurrent import LSTM

    init_style = params["init_style"]
    cs = params["conv_size"]

    # nb_filters = 32
    nb_filters = [32, 64, 128]
    kernel_size = (cs, cs)
    img_rows, img_cols = params["patch_size"][0], params["patch_size"][1]
    nb_pool = 2
    p = 0.5
    final_activation = params['final_activation']
    flat_output = params['flat_output']
    learning_rate = params['learning_rate']
    optimizer = params['optimizer']
    loss = params['loss']

    input_shape = (10, params["num_channels"], img_rows, img_cols)

    input = Input(shape=input_shape)
    l1 = LSTMConv2D(64,3,3,dim_ordering='tf',border_mode='same', return_sequences=True)(input)

    model = Model(input=input, output=l1)

    model.compile(loss=loss, optimizer='rmsprop')

    return model



def unet_3d_generator(convs_per_layer=2, tile_size=None, channels=None, loss='mean_squared_error', init_style='glorot_uniform',
         internal_dropout_rate=0, final_dropout_rate=0.5, final_activation='sigmoid', flat_output=False,
         learning_rate=1e-4, batch_norm=False, compile=True, input_layer=None, reporter_loss=None, filters=None, **kwargs):

    #
    # img_dim = tile_size
    #
    # img_rows, img_cols, img_depth = tile_size[0], tile_size[1], tile_size[2]
    # num_channels = len(channels)
    #
    # if K.image_data_format() == 'channels_first':
    #     input_shape = (num_channels, img_rows, img_cols, img_depth)
    #     concat_axis = -3
    # elif K.image_data_format() == 'channels_last':
    #     input_shape = (img_rows, img_cols, img_depth, num_channels)
    #     concat_axis = -1
    #
    # input = Input(input_shape)
    # conv1 = Convolution3D(nb_filters, 3, 3, 3, border_mode='same')(input)
    # conv1 = BatchNormalization()(conv1)
    # conv1 = LeakyReLU(0.2)(conv1)
    # down1 = MaxPooling3D((2, 2, 1))(conv1)
    #
    # conv2 = Convolution3D(nb_filters*2, 3, 3, 3, border_mode='same')(down1)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = LeakyReLU(0.2)(conv2)
    # down2 = MaxPooling3D((2, 2, 1))(conv2)
    #
    # conv3 = Convolution3D(nb_filters*2, 3, 3, 3, border_mode='same')(down2)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = LeakyReLU(0.2)(conv3)
    # down3 = MaxPooling3D((2, 2, 1))(conv3)
    #
    # conv4 = Convolution3D(nb_filters*4, 3, 3, 3, border_mode='same')(down3)
    # conv4 = BatchNormalization()(conv4)
    # conv4 = LeakyReLU(0.2)(conv4)
    #
    # up1 = merge([UpSampling3D((2, 2, 1))(conv4), conv3], concat_axis=concat_axis, mode='concat')
    # conv5 = Convolution3D(nb_filters*2, 3, 3, 3, border_mode='same')(up1)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = LeakyReLU(0.2)(conv5)
    #
    # up2 = merge([UpSampling3D((2, 2, 1))(conv5), conv2], concat_axis=concat_axis, mode='concat')
    # conv6 = Convolution3D(nb_filters, 3, 3, 3, border_mode='same')(up2)
    # conv6 = BatchNormalization()(conv6)
    # conv6 = LeakyReLU(0.2)(conv6)
    #
    # up3 = merge([UpSampling3D((2, 2, 1))(conv6), conv1], concat_axis=concat_axis, mode='concat')
    # conv7 = Convolution3D(nb_filters, 3, 3, 3, border_mode='same')(up3)
    # conv7 = BatchNormalization()(conv7)
    # conv7 = LeakyReLU(0.2)(conv7)
    #
    # conv5 = Convolution3D(1, 1, 1, 1)(conv7)
    # act1 = Activation('sigmoid')(conv5)
    # model = Model(input=input, output=act1)
    #
    # return model
    from keras.layers import Deconvolution2D, Conv3D, Conv2DTranspose, Concatenate

    KERAS_2 = True

    def Convolution(f, k=3, s=2, border_mode='same', is_3d=False, **kwargs):
        """Convenience method for Convolutions."""
        if is_3d:
            return Conv3D(f, kernel_size=(k, k, k), strides=(s, s, 1), border_mode=border_mode)
        else:
            if KERAS_2:
                return Convolution2D(f,
                                     kernel_size=(k, k),
                                     padding=border_mode,
                                     strides=(s, s),
                                     **kwargs)
            else:
                return Convolution2D(f, k, k, border_mode=border_mode,
                                     subsample=(s, s),
                                     **kwargs)

    def Deconvolution(i, f, output_shape, k=2, s=2, is_3d=False, **kwargs):
        """Convenience method for Transposed Convolutions."""

        if is_3d:
            l = UpSampling3D((s, s, 1))(i)
            l = Conv3D(f, kernel_size=(k, k, k), padding='same')(l)
            return l
        else:
            if KERAS_2:
                return Conv2DTranspose(f,
                                       kernel_size=(k, k),
                                       output_shape=output_shape,
                                       strides=(s, s),
                                       data_format=K.image_data_format(),
                                       **kwargs)(i)
            else:
                return Deconvolution2D(f, k, k, output_shape=output_shape,
                                       subsample=(s, s), **kwargs)(i)

    def BatchNorm(mode=2, axis=-1, **kwargs):
        """Convenience method for BatchNormalization layers."""
        if KERAS_2:
            return BatchNormalization(axis=axis, **kwargs)
        else:
            return BatchNormalization(mode=2, axis=axis, **kwargs)

    def concatenate_layers(inputs, concat_axis, mode='concat'):
        if KERAS_2:
            assert mode == 'concat', "Only concatenation is supported in this wrapper"
            return Concatenate(axis=concat_axis)(inputs)
        else:
            return merge(inputs=inputs, concat_axis=concat_axis, mode=mode)

    dropout_rate = 0.5

    batch_size=1

    merge_params = {
        'mode': 'concat',
        'concat_axis': -1
    }

    def get_deconv_shape(samples, channels, x_dim, y_dim):
        return samples, x_dim, y_dim, channels

    if tile_size is None:
        tile_size = [512, 512]

    if channels is None:
        channels = [0, ]

    img_rows, img_cols, depth = tile_size[0], tile_size[1], tile_size[2]
    num_channels = len(channels)

    if K.image_data_format() == 'channels_first':
        input_shape = (num_channels, img_rows, img_cols, depth)
        concat_axis = -3
    elif K.image_data_format() == 'channels_last':
        input_shape = (img_rows, img_cols, depth, num_channels)
        concat_axis = -1

    if input_layer is not None:
        inputs = input_layer
    else:
        inputs = Input(input_shape)


    is_3d = True
    nf = 32

    # nf*2 x 128 x 128
    conv3 = Convolution(nf, s=1, is_3d=is_3d)(inputs)
    conv3 = Convolution(nf, is_3d=is_3d)(conv3)
    #conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    #x = Dropout(dropout_rate)(x)
    # nf*4 x 64 x 64

    conv4 = Convolution(nf * 2, s=1, is_3d=is_3d)(x)
    conv4 = Convolution(nf * 2, is_3d=is_3d)(conv4)
    #conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    #x = Dropout(dropout_rate)(x)
    # nf*8 x 32 x 32

    conv5 = Convolution(nf * 4,s=1, is_3d=is_3d)(x)
    conv5 = Convolution(nf * 4, is_3d=is_3d)(conv5)
    #conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    #x = Dropout(dropout_rate)(x)
    # nf*8 x 16 x 16

    conv6 = Convolution(nf * 8,s=1, is_3d=is_3d)(x)
    conv6 = Convolution(nf * 8, is_3d=is_3d)(conv6)
    #conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    #x = Dropout(dropout_rate)(x)
    # nf*8 x 8 x 8

    conv7 = Convolution(nf * 16,s=1, is_3d=is_3d)(x)
    conv7 = Convolution(nf * 16, is_3d=is_3d)(conv7)
    #conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    #x = Dropout(dropout_rate)(x)
    # nf*8 x 4 x 4


    dconv2 = Deconvolution(x, nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 4, 4), s=1,is_3d=is_3d)
    dconv2 = Deconvolution(dconv2, nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 4, 4), is_3d=is_3d)
    #dconv2 = BatchNorm()(dconv2)
    #dconv2 = Dropout(0.5)(dconv2)
    x = concatenate_layers([dconv2, conv6], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(x, nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 8, 8), s=1, is_3d=is_3d)
    dconv3 = Deconvolution(dconv3, nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 8, 8), is_3d=is_3d)
    #dconv3 = BatchNorm()(dconv3)
    #dconv3 = Dropout(0.5)(dconv3)
    x = concatenate_layers([dconv3, conv5], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(x, nf * 4,
                           get_deconv_shape(batch_size, nf * 8, 16, 16), s=1, is_3d=is_3d)
    dconv4 = Deconvolution(dconv4, nf * 4,
                           get_deconv_shape(batch_size, nf * 8, 16, 16), is_3d=is_3d)
    #dconv4 = BatchNorm()(dconv4)
    x = concatenate_layers([dconv4, conv4], **merge_params)
    x = LeakyReLU(0.2)(x)
    #x = Dropout(dropout_rate)(x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(x, nf * 2,
                           get_deconv_shape(batch_size, nf * 8, 32, 32), s=1, is_3d=is_3d)
    dconv5 = Deconvolution(dconv5, nf * 2,
                           get_deconv_shape(batch_size, nf * 8, 32, 32), is_3d=is_3d)
    #dconv5 = BatchNorm()(dconv5)
    x = concatenate_layers([dconv5, conv3], **merge_params)
    x = LeakyReLU(0.2)(x)
    #x = Dropout(dropout_rate)(x)
    # nf*(8 + 8) x 32 x 32




    dconv9 = Deconvolution(x, 1,
                           get_deconv_shape(batch_size, 1, 128, 128), s=1, is_3d=is_3d)
    dconv9 = Deconvolution(dconv9, 1,
                           get_deconv_shape(batch_size, 1, 128, 128), is_3d=is_3d)
    # out_ch x 128 x 128


    act = 'sigmoid'
    out = Activation(act)(dconv9)

    unet = Model(inputs, out)
    unet.compile(loss=get_loss(loss), optimizer=Adam(lr=1e-4))

    return unet
