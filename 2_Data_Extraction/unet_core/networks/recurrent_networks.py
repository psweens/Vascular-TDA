import numpy as np

from keras.layers import Input, Lambda,Conv2D, Dropout, TimeDistributed, Reshape, BatchNormalization, Bidirectional, Activation, MaxPooling2D, Deconvolution2D, Permute, UpSampling2D
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam
from skimage.transform import pyramid_expand
from keras.layers.convolutional_recurrent import ConvLSTM2D
# from unet_core.networks.conv_lstm import ConvLSTM2D
from unet_core.networks.layers import GumbelSkeletonLayer
from unet_core.networks.convolutional_networks import get_loss
import keras.backend as K



class RollingArray(object):

    def __init__(self, data):
        self.data = data

    def push_back(self, data_point, axis_roll=0):
        self.data = np.roll(self.data,-1,axis=axis_roll)
        self.data[-1, :, :] = data_point

    def append(self, data_point, axis=0):
        self.data = np.concatenate((self.data, data_point.reshape((1, data_point.shape[0], -1))), axis=axis)

    def yield_last(self, num_points=10):

        if num_points > 0:
            return self.data[-num_points:, :, :]
        else:
            return self.data


def BidirectionalLSTMConv2D(input_layer, nb_filter=20, activation=True, name=None, **kwargs):

    if K.image_data_format() == 'channels_first':
        input_layer = Permute((1, 3, 4, 2))(input_layer)

    if name is not None:
        name1 = name + '_forward'
        name2 = name + '_backward'
    else:
        name1 = None
        name2 = None

    seq1 = ConvLSTM2D(kernel_size=(3, 3),
                      go_backwards=False,
                      filters=nb_filter,
                      return_sequences=True,
                      padding="same",
                      data_format='channels_last',
                      name=name1)(input_layer)

    if K.image_data_format() == 'channels_first':
        seq1 = Permute((1, 4, 2, 3))(seq1)

    seq2 = ConvLSTM2D(kernel_size=(3, 3),
                      go_backwards=True,
                      filters=nb_filter,
                      padding="same",
                      return_sequences=True, name=name2)(input_layer)
    seq2 = Lambda(lambda x: K.reverse(x, axes=1), output_shape=lambda x: x)(seq2)

    if K.image_data_format() == 'channels_first':
        merge_axis = -3
    else:
        merge_axis = -1

    merged = Concatenate(axis=merge_axis)([seq1, seq2])

    return merged


def skipConvLSTMLayer(input,rows,cols, nb_filters=20, **kwargs):

    branch1 = Bidirectional(ConvLSTM2D(kernel_size=(3, 3), go_backwards=True, filters=nb_filters, return_sequences=True,
                                       data_format=None, padding="same"),
                            input_shape=(None, 1, rows, cols), merge_mode='sum')(input)
    branch1 = BatchNormalization(axis=-3)(branch1)

    branch2 = TimeDistributed(Convolution2D(1, 1, 1))(input)

    # layer1 = merge([branch1, branch2], concat_axis=-3, mode='concat')

    layer1 = Concatenate(axis=-1)([branch1, branch2])

    return layer1


def skipConvLSTMLayer2(input,rows,cols, nb_filters=20, merge_mode='concat', first_layer=False, **kwargs):

    if first_layer:
        branch1 = ConvLSTM2D(nb_filter=nb_filters, nb_row=3, nb_col=3,
                             border_mode="same", return_sequences=True, dim_ordering='th',
                             input_shape=(None, 1, rows, cols), go_backwards=False)(input)
        branch2 = ConvLSTM2D(nb_filter=nb_filters, nb_row=3, nb_col=3,
                             border_mode="same", return_sequences=True, dim_ordering='th',
                             input_shape=(None, 1, rows, cols), go_backwards=True)(input)
    else:
        branch1 = ConvLSTM2D(nb_filter=nb_filters, nb_row=3, nb_col=3,
                             border_mode="same", return_sequences=True, dim_ordering='th', go_backwards=False)(input)
        branch2 = ConvLSTM2D(nb_filter=nb_filters, nb_row=3, nb_col=3,
                             border_mode="same", return_sequences=True, dim_ordering='th', go_backwards=True)(input)

    lstm1 = merge([branch1, branch2], concat_axis=-3, mode=merge_mode)
    lstm1 = TimeDistributed(Convolution2D(1, 1, 1))(lstm1)

    layer1 = merge([lstm1, input], concat_axis=-3, mode='concat')

    layer1 = BatchNormalization(axis=-3)(layer1)
    layer1 = LeakyReLU(alpha=0.1)(layer1)

    return layer1


def ConvLSTMLayer(input, nb_filters=20,
                  merge_mode='concat',
                  stateful=False,
                  bidirectional=True,
                  batch_norm=False,
                  activation=True,
                  **kwargs):

    if bidirectional:
        lstm1 = BidirectionalLSTMConv2D(input,
                                        nb_filter=nb_filters,
                                        return_sequences=True,
                                        merge_mode=merge_mode,
                                        **kwargs)
    else:
        lstm1 = ConvLSTM2D(kernel_size=(3, 3), filters=nb_filters, return_sequences=True, padding="same")(input)

    # if batch_norm:
    #     if K.image_data_format() == 'channels_first':
    #         lstm1 = BatchNormalization(axis=-3)(lstm1)
    #     elif  K.image_data_format() == 'channels_last':
    #         lstm1 = BatchNormalization(axis=-1)(lstm1)

    # lstm1 = TimeDistributed(Conv2D(1, (1, 1)))(lstm1)
    # """LeakyReLU used to avoid the Dying ReLU problem
    # (http://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks)"""
    # lstm1 = LeakyReLU(alpha=0.1)(lstm1)

    return lstm1


def miniSkipConvLSTMModel(tile_size=None, learning_rate=0.001, **kwargs):
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input((None, 1, rows, cols))
    layer1 = skipConvLSTMLayer2(layer0, rows, cols, **kwargs)
    layer2 = skipConvLSTMLayer2(layer1, rows, cols, **kwargs)

    conv = TimeDistributed(Convolution2D(1,1,1,activation='linear'))(layer2)

    seq = Model(input=layer0, output=conv)

    seq.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate))

    return seq

def miniConvLSTMModel(tile_size=None, learning_rate=0.001, **kwargs):
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input((None, 1, rows, cols))
    layer1 = ConvLSTMLayer(layer0, rows, cols, **kwargs)

    # conv = TimeDistributed(Convolution2D(1,1,1,activation='linear'))(layer1)

    seq = Model(input=layer0, output=layer1)
    seq.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate))

    return seq


def miniConvLSTMModel_skeleton(tau=2.0, tile_size=None, learning_rate=0.001, loss='gumbel_sample_bce', **kwargs):
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input((None, 1, rows, cols))
    layer1 = ConvLSTMLayer(layer0, rows, cols, **kwargs)

    K_tau = K.variable(tau, name='temparature')
    sk1 = TimeDistributed(GumbelSkeletonLayer(K_tau))(layer1)

    seq = Model(input=layer0, output=sk1)
    seq.compile(loss=get_loss(loss), optimizer=RMSprop(lr=learning_rate))
    seq.tau = K_tau
    return seq


def miniConvLSTMModel_stateful(tile_size=None, learning_rate=0.001, seq_len=10, **kwargs):
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input(batch_shape=(1, seq_len, 1, rows, cols))
    layer1 = ConvLSTMLayer(layer0, rows, cols, **kwargs)

    conv = TimeDistributed(Convolution2D(1,1,1,activation='linear'))(layer1)

    seq = Model(input=layer0, output=conv)

    seq.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate))

    return seq


def skipConvLSTMModel(tile_size=None, learning_rate=0.001, **kwargs):
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input((None, 1, rows, cols))
    layer1 = skipConvLSTMLayer2(layer0, rows, cols, **kwargs)
    layer2 = skipConvLSTMLayer2(layer1, rows, cols, **kwargs)
    layer3 = skipConvLSTMLayer2(layer2, rows, cols, **kwargs)
    layer4 = skipConvLSTMLayer2(layer3, rows, cols, **kwargs)

    conv = TimeDistributed(Convolution2D(1,1,1,activation='linear'))(layer4)

    seq = Model(input=layer0, output=conv)

    seq.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate))

    return seq


def ConvLSTMModel(tile_size=None, learning_rate=0.001, final_activation='sigmoid', loss='mean_squared_error', **kwargs):
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input((None, 1, rows, cols))
    layer1 = ConvLSTMLayer(layer0, rows, cols, **kwargs)
    layer2 = ConvLSTMLayer(layer1, rows, cols, **kwargs)

    conv = TimeDistributed(Convolution2D(1,1,1,activation=final_activation))(layer2)

    seq = Model(input=layer0, output=conv)
    seq.compile(loss=get_loss(loss), optimizer=RMSprop(lr=learning_rate))

    return seq

def ConvLSTMModel_skeleton(tau=2.0, tile_size=None, learning_rate=0.001, final_activation='sigmoid', loss='gumbel_sample_bce', **kwargs):
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input((None, 1, rows, cols))
    layer1 = ConvLSTMLayer(layer0, rows, cols, **kwargs)
    layer2 = ConvLSTMLayer(layer1, rows, cols, **kwargs)

    K_tau = K.variable(tau, name='temparature')
    sk1 = TimeDistributed(GumbelSkeletonLayer(K_tau))(layer2)

    seq = Model(input=layer0, output=sk1)
    seq.compile(loss=get_loss(loss), optimizer=RMSprop(lr=learning_rate))
    seq.tau = K_tau
    return seq

def DeepConvLSTMModel(tile_size=None, learning_rate=0.001, compile=True, **kwargs):

    nb_filters = 64
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input((None, rows, cols, 1))
    layer1 = ConvLSTMLayer(layer0, rows, cols, **kwargs)
    layer2 = ConvLSTMLayer(layer1, rows, cols, **kwargs)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), data_format=None, padding="same"))(layer2)
    layer3 = ConvLSTMLayer(pool1, rows, cols, **kwargs)
    layer4 = ConvLSTMLayer(layer3, rows, cols, **kwargs)
    deconv1 = TimeDistributed(Deconvolution2D(nb_filters, 3, 3, output_shape=(None, None, rows, cols), subsample=(2,2),activation='relu', dim_ordering='default', border_mode='same'))(layer4)
    layer5 = ConvLSTMLayer(deconv1, rows, cols, **kwargs)
    conv = TimeDistributed(Conv2D(1, (1,1),activation='linear'))(layer5)

    seq = Model(input=layer0, output=conv)
    if compile:
        seq.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate))

    return seq

def DeepUNetLSTMModel(tile_size=None, learning_rate=0.001, loss='mean_squared_error', **kwargs):
    from keras.layers.merge import concatenate
    nb_filters = 10
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input((None, rows, cols, 1))
    layer1 = ConvLSTMLayer(layer0, rows, cols, **kwargs)
    layer2 = ConvLSTMLayer(layer1, rows, cols, **kwargs)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='default'))(layer2)
    layer3 = ConvLSTMLayer(pool1, rows, cols, **kwargs)
    layer4 = ConvLSTMLayer(layer3, rows, cols, **kwargs)
    deconv1 = TimeDistributed(UpSampling2D(size=(2, 2), dim_ordering='default'))(layer4)
    merge1 = concatenate([deconv1, layer2], axis=-1)
    layer5 = ConvLSTMLayer(merge1, rows, cols, **kwargs)
    conv = TimeDistributed(Conv2D(1, 1, 1, activation='linear'))(layer5)

    seq = Model(input=layer0, output=conv)
    seq.compile(loss=get_loss(loss), optimizer=RMSprop(lr=learning_rate))

    return seq

def DeepUNetLSTMModel_skeleton(tau=2.0, tile_size=None, learning_rate=0.001, loss='mean_squared_error', **kwargs):
    nb_filters = 10
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input((None, 1, rows, cols))
    layer1 = ConvLSTMLayer(layer0, rows, cols, **kwargs)
    layer2 = ConvLSTMLayer(layer1, rows, cols, **kwargs)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='default'))(layer2)
    layer3 = ConvLSTMLayer(pool1, rows, cols, **kwargs)
    layer4 = ConvLSTMLayer(layer3, rows, cols, **kwargs)
    deconv1 = TimeDistributed(UpSampling2D(size=(2, 2), dim_ordering='default'))(layer4)
    merge1 = merge([deconv1, layer2], concat_axis=-3, mode='concat')
    layer5 = ConvLSTMLayer(merge1, rows, cols, **kwargs)
    conv = TimeDistributed(Convolution2D(1, 1, 1, activation='sigmoid'))(layer5)

    K_tau = K.variable(tau, name='temparature')
    sk1 = TimeDistributed(GumbelSkeletonLayer(K_tau))(conv)

    seq = Model(input=layer0, output=sk1)
    seq.compile(loss=get_loss(loss), optimizer=RMSprop(lr=learning_rate))
    seq.tau = K_tau
    return seq

def DeepUNetLSTMModel_stateful(tile_size=None, learning_rate=0.001, seq_len=10, **kwargs):
    nb_filters = 10
    rows, cols = tile_size[0], tile_size[1]

    layer0 = Input(batch_shape=(1, seq_len, 1, rows, cols))
    layer1 = ConvLSTMLayer(layer0, rows, cols, **kwargs)
    layer2 = ConvLSTMLayer(layer1, rows, cols, **kwargs)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='default'))(layer2)
    layer3 = ConvLSTMLayer(pool1, rows, cols, **kwargs)
    layer4 = ConvLSTMLayer(layer3, rows, cols, **kwargs)
    # deconv1 = TimeDistributed(Deconvolution2D(nb_filters, 3, 3, output_shape=(None, None, rows, cols), subsample=(2, 2), activation='linear',
    #                     dim_ordering='th', border_mode='same'))(layer4)
    deconv1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), dim_ordering='default', border_mode='same'))(layer4)
    merge1 = merge([deconv1, layer2], concat_axis=-3, mode='concat')
    layer5 = ConvLSTMLayer(merge1, rows, cols, **kwargs)
    conv = TimeDistributed(Convolution2D(1, 1, 1, activation='linear'))(layer5)

    seq = Model(input=layer0, output=conv)
    seq.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate))

    return seq

def td_unet_sub_unit(in_layer, layer_count, nb_filters=32, init_style='glorot_uniform', dropout_rate=0.1, batch_norm=False, leaky_activation=False,
             filter_size=3, **kwargs):

    name_conv = 'conv2d_{}'.format(layer_count)
    name_bn = 'batchnormalization_{}'.format(layer_count)
    name_dropout = 'dropout_{}'.format(layer_count)

    if K.image_data_format() == 'channels_first':
        bn_axis = -3
    elif K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('invalid image_data_format(): {}'.format(K.image_data_format()))

    cs = filter_size
    c1 = TimeDistributed(Conv2D(nb_filters, kernel_size=(cs, cs), kernel_initializer=init_style, padding='same'), name=name_conv)(in_layer)
    if batch_norm:
        # c1 = TimeDistributed(BatchNormalization(axis=bn_axis), name=name_bn)(c1)
        c1 = BatchNormalization(axis=bn_axis, name=name_bn)(c1)

    if leaky_activation:
        c1 = LeakyReLU(0.1)(c1)
    else:
        c1 = Activation('relu')(c1)

    if dropout_rate > 0.0:
        # c1 = TimeDistributed(Dropout(dropout_rate), name=name_dropout)(c1)
        c1 = Dropout(dropout_rate, name=name_dropout)(c1)

    layer_count += 1
    return c1, layer_count


def td_unet_unit(input, layer_count, nb_layers=2, nb_filters=32, **kwargs):

    conv1, layer_count = td_unet_sub_unit(input, layer_count, nb_filters=nb_filters, **kwargs)
    for i in range(nb_layers-1):
        conv1, layer_count = td_unet_sub_unit(conv1, layer_count, nb_filters=nb_filters, **kwargs)

    return conv1, layer_count


def TimeDistributedUnet(convs_per_layer=2, tile_size=None, channels=[0,], loss='mean_squared_error', init_style='glorot_uniform',
         internal_dropout_rate=0, final_dropout_rate=0.5, final_activation='sigmoid', flat_output=False,
         learning_rate=1e-4, batch_norm=False, compile=True, input_layer=None, filters=None, **kwargs):

    if tile_size is None:
        tile_size = [512, 512]

    cs = 3
    rows, cols = tile_size[0], tile_size[1]
    nb_channels = len(channels)

    if K.image_data_format() == 'channels_first':
        input_shape = (None, nb_channels, rows, cols)
        concat_axis = -3
    elif K.image_data_format() == 'channels_last':
        input_shape = (None, rows, cols, nb_channels)
        concat_axis = -1
    else:
        raise ValueError('invalid image_data_format(): {}'.format(K.image_data_format()))

    if input_layer is not None:
        inputs = input_layer
    else:
        inputs = Input(input_shape)

    if filters is None:
        filters = [32, 32, 32, 32, 32, 32, 32, 32, 32]

    layer_count = 20


    conv1, layer_count = td_unet_unit(inputs, layer_count, nb_filters=filters[0], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name='maxpool2d_1')(conv1)

    conv2, layer_count = td_unet_unit(pool1, layer_count, nb_filters=filters[1], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name='maxpool2d_2')(conv2)

    conv3, layer_count = td_unet_unit(pool2, layer_count, nb_filters=filters[2], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name='maxpool2d_3')(conv3)

    conv4, layer_count = td_unet_unit(pool3, layer_count, nb_filters=filters[3], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name='maxpool2d_4')(conv4)

    conv5, layer_count = td_unet_unit(pool4, layer_count, nb_filters=filters[4], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate)

    up6 = Concatenate(axis=concat_axis)([TimeDistributed(UpSampling2D(size=(2, 2)), name='upsampling2d_1')(conv5), conv4])
    conv6, layer_count = td_unet_unit(up6, layer_count, nb_filters=filters[5], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                          dropout_rate=internal_dropout_rate)

    up7 = Concatenate(axis=concat_axis)([TimeDistributed(UpSampling2D(size=(2, 2)), name='upsampling2d_2')(conv6), conv3])
    conv7, layer_count = td_unet_unit(up7, layer_count, nb_filters=filters[6], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate)

    up8 = Concatenate(axis=concat_axis)([TimeDistributed(UpSampling2D(size=(2, 2)), name='upsampling2d_3')(conv7), conv2])
    conv8, layer_count = td_unet_unit(up8, layer_count, nb_filters=filters[7], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate)

    up9 = Concatenate(axis=concat_axis)([TimeDistributed(UpSampling2D(size=(2, 2)), name='upsampling2d_4')(conv8), conv1])
    conv9, layer_count = td_unet_unit(up9, layer_count, nb_filters=filters[8], filter_size=3, nb_layers=convs_per_layer, init_style=init_style, batch_norm=batch_norm,
                      dropout_rate=internal_dropout_rate)

    if final_dropout_rate > 0.0:
        conv9 = TimeDistributed(Dropout(final_dropout_rate), name='final_dropout')(conv9)

    # conv9 = TimeDistributed(Conv2D(1, (1, 1)), name='convolution2d_{}'.format(layer_count))(conv9)
    # conv9 = Activation(final_activation)(conv9)

    return conv9


def FullUNetCNNConvLSTM(tile_size=None, learning_rate=0.001, final_activation='sigmoid', loss='binary_crossentropy',
                        reporter_loss='binary_crossentropy', channels=None, compile=True, seq_len=None, **kwargs):

    if channels is None:
        channels = (2,1)

    rows, cols = tile_size[0], tile_size[1]
    nb_channels = len(channels)

    if K.image_data_format() == 'channels_first':
        input_shape = (seq_len, nb_channels, rows, cols)
        concat_axis = -3
    elif K.image_data_format() == 'channels_last':
        input_shape = (seq_len, rows, cols, nb_channels)
        concat_axis = -1
    else:
        raise ValueError('invalid image_data_format(): {}'.format(K.image_data_format()))

    inputs = Input(input_shape)

    unet0 = TimeDistributedUnet(input_layer=inputs, tile_size=tile_size, **kwargs)

    layer1 = ConvLSTMLayer(unet0, nb_filters=32, **kwargs)

    conv = TimeDistributed(Conv2D(1, (1,1), name='regular'))(layer1)
    act = Activation(final_activation, name='regular')(conv)

    reporter = Activation(final_activation,name='reporter')(conv)

    if reporter_loss is not None:
        seq = Model(input=inputs, output=[act, reporter])
        if compile:
            seq.compile(loss=[get_loss(loss), get_loss(reporter_loss)], optimizer=Adam(lr=learning_rate), loss_weights=[1, 0])
    else:
        seq = Model(inputs=inputs, outputs=act)
        if compile:
            seq.compile(loss=get_loss(loss), optimizer=Adam(lr=learning_rate))


    return seq

def DeepUNetCNNConvLSTM(tile_size=None, learning_rate=0.001, final_activation='sigmoid', loss='binary_crossentropy',
                        reporter_loss='binary_crossentropy', channels=None, compile=True, **kwargs):

    if channels is None:
        channels = (2,1)

    rows, cols = tile_size[0], tile_size[1]
    nb_channels = len(channels)

    if K.image_data_format() == 'channels_first':
        input_shape = (None, nb_channels, rows, cols)
        concat_axis = -3
    elif K.image_data_format() == 'channels_last':
        input_shape = (None, rows, cols, nb_channels)
        concat_axis = -1
    else:
        raise ValueError('invalid image_data_format(): {}'.format(K.image_data_format()))

    inputs = Input(input_shape)

    unet0 = TimeDistributedUnet(input_layer=inputs, tile_size=tile_size, **kwargs)

    layer1 = ConvLSTMLayer(unet0, nb_filters=32, name='bonus_layer1', **kwargs)
    # layer1 = TimeDistributed(Conv2D(1, (1,1), name='bonus_conv1'))(layer1)
    layer2 = ConvLSTMLayer(layer1, nb_filters=32, name='bonus_layer2', **kwargs)
    # layer2 = TimeDistributed(Conv2D(1, (1,1), name='bonus_conv2'))(layer2)
    layer3 = ConvLSTMLayer(layer2, nb_filters=32, name='bonus_layer3', **kwargs)
    # layer3 = TimeDistributed(Conv2D(1, (1,1), name='bonus_conv3'))(layer3)
    layer4 = ConvLSTMLayer(layer3, nb_filters=32, name='bonus_layer4', **kwargs)

    conv = TimeDistributed(Conv2D(1, (1,1), name='regular'))(layer4)
    act = Activation(final_activation, name='regular')(conv)

    reporter = Activation(final_activation,name='reporter')(conv)

    if reporter_loss is not None:
        seq = Model(input=inputs, output=[act, reporter])
        if compile:
            seq.compile(loss=[get_loss(loss), get_loss(reporter_loss)], optimizer=Adam(lr=learning_rate), loss_weights=[1, 0])
    else:
        seq = Model(inputs=inputs, outputs=act)
        if compile:
            seq.compile(loss=get_loss(loss), optimizer=Adam(lr=learning_rate))


    return seq

def ShallowUNetCNNConvLSTM(tile_size=None, learning_rate=0.001, final_activation='sigmoid', loss='binary_crossentropy',
                        reporter_loss='binary_crossentropy', channels=None, compile=True, **kwargs):

    if channels is None:
        channels = (2,1)

    rows, cols = tile_size[0], tile_size[1]
    nb_channels = len(channels)

    if K.image_data_format() == 'channels_first':
        input_shape = (None, nb_channels, rows, cols)
        concat_axis = -3
    elif K.image_data_format() == 'channels_last':
        input_shape = (None, rows, cols, nb_channels)
        concat_axis = -1
    else:
        raise ValueError('invalid image_data_format(): {}'.format(K.image_data_format()))

    inputs = Input(input_shape)

    unet0 = TimeDistributedUnet(input_layer=inputs, tile_size=tile_size, **kwargs)

    layer1 = ConvLSTMLayer(unet0, nb_filters=32, **kwargs)

    conv = TimeDistributed(Conv2D(1, (1,1), name='regular'))(layer1)
    act = Activation(final_activation, name='regular')(conv)

    reporter = Activation(final_activation,name='reporter')(conv)

    if reporter_loss is not None:
        seq = Model(input=inputs, output=[act, reporter])
        if compile:
            seq.compile(loss=[get_loss(loss), get_loss(reporter_loss)], optimizer=Adam(lr=learning_rate), loss_weights=[1, 0])
    else:
        seq = Model(inputs=inputs, outputs=act)
        if compile:
            seq.compile(loss=get_loss(loss), optimizer=Adam(lr=learning_rate))


    return seq


def ChanUNetCNNConvLSTM(tile_size=None, learning_rate=0.001, final_activation='sigmoid', loss='binary_crossentropy',
                        reporter_loss='binary_crossentropy', channels=None, compile=True, **kwargs):

    if channels is None:
        channels = (2,1)

    rows, cols = tile_size[0], tile_size[1]
    nb_channels = len(channels)

    if K.image_data_format() == 'channels_first':
        input_shape = (None, nb_channels, rows, cols)
        concat_axis = -3
    elif K.image_data_format() == 'channels_last':
        input_shape = (None, rows, cols, nb_channels)
        concat_axis = -1
    else:
        raise ValueError('invalid image_data_format(): {}'.format(K.image_data_format()))

    inputs = Input(input_shape)

    unet0 = TimeDistributedUnet(input_layer=inputs, tile_size=tile_size, **kwargs)

    layer1 = ConvLSTMLayer(unet0, nb_filters=32, **kwargs)
    layer2 = ConvLSTMLayer(layer1, nb_filters=32, **kwargs)
    pool0 = TimeDistributed(MaxPooling2D((2,2)))(layer2)
    layer3 = ConvLSTMLayer(pool0, nb_filters=32, **kwargs)
    layer4 = ConvLSTMLayer(layer3, nb_filters=32, **kwargs)
    pool1 = TimeDistributed(UpSampling2D((2,2)))(layer4)
    layer5 = ConvLSTMLayer(pool1, nb_filters=32, **kwargs)
    layer6 = ConvLSTMLayer(layer5, nb_filters=32, **kwargs)

    conv = TimeDistributed(Conv2D(1, (1,1), name='regular'))(layer6)
    act = Activation(final_activation, name='regular')(conv)

    reporter = Activation(final_activation,name='reporter')(conv)

    if reporter_loss is not None:
        seq = Model(input=inputs, output=[act, reporter])
        if compile:
            seq.compile(loss=[get_loss(loss), get_loss(reporter_loss)], optimizer=Adam(lr=learning_rate), loss_weights=[1, 0])
    else:
        seq = Model(inputs=inputs, outputs=act)
        if compile:
            seq.compile(loss=get_loss(loss), optimizer=Adam(lr=learning_rate))


    return seq