from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import print_function

import math
import os.path as path
import sys
import threading

import nibabel as nib
import scipy.ndimage as ndi
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from numpy.random import rand, permutation
from scipy import linalg
from scipy.signal import convolve2d
from skimage.filters import gaussian

from unet_core.networks.convolutional_networks import _get_network
from unet_core.utils.data_utils import im_smooth, TwoImageIterator, TwoImageCrossValidateIterator


from unet_core.networks.callbacks import *
from keras.callbacks import ModelCheckpoint
import os
from unet_core.core import _get_model, _default_params, _parse_params
from unet_core.utils.data_utils import get_data_from_params

smooth = 1.


class epoch_print_callback(Callback):
    def __init__(self):
        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        print("Completed Epoch %d: Current loss = %f"%(epoch, current))



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def train_model_anneal(input_params=None, annealing_rate=0.005, min_temp=0.05, x=None, y=None):
    if input_params is None:
        input_params = {}

    input_params = _parse_params(input_params)

    # print('Training on device: {}'.format(theano.config.device))
    # print('cuDNN version: {}'.format(theano.sandbox.cuda.dnn_version()[1]))

    params = _default_params()
    params.update(input_params)
    model_function, model_params = _get_model(params['architecture'])
    params.update(model_params)
    params.update(input_params)
    params['image_dim_ordering'] = K.image_dim_ordering()

    if x is None or y is None:
        x, y = get_data_from_params(params)
        x, y = _preprocess_data(x, y, params)

    model = model_function(**params)

    if params['initial_weights'] is not None:
        print('loading weights from: {}'.format(params['initial_weights']))
        model.load_weights(params['initial_weights'])

    output_folder = os.path.join(params['output_folder'], params['experiment_name']) + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plot_loss_callback = OutputHistory(output_folder + 'loss.png', output_folder+ 'loss.json',
                                       save_graphs=params['save_graphs'], plot_logs=params['plot_logs'])

    model_checkpoint = ModelCheckpoint(output_folder + 'weights.{epoch:02d}-{val_loss:.6f}.hdf5', save_best_only=False, save_weights_only=True)

    summary(params, x, y)
    json.dump(params, open(output_folder+'params.json', 'w'), sort_keys=True, indent=4)

    if params['augmentation']:
        train_datagen, x_valid, y_valid = _make_training_generators(x, y, params)
        model.fit_generator(train_datagen, samples_per_epoch=train_datagen.X.shape[0], nb_epoch=params['nb_epoch'], verbose=params["verbose"],
                            callbacks=[plot_loss_callback, model_checkpoint], validation_data=(x_valid, y_valid))
    else:
        if params['pad_target_features'] is not None:
            y = np.concatenate([y,]*params['pad_target_features'], axis=-3)

        for e in range(params['nb_epoch']):
            print(model.tau.eval())
            model.fit(x, y, epochs=e+1, callbacks=[plot_loss_callback, model_checkpoint],
                  batch_size=params['batch_size'], verbose=params["verbose"], shuffle=True,
                  validation_split=params['validation_split'], initial_epoch=e)

            K.set_value(model.tau, np.max([K.get_value(model.tau) * np.exp(-annealing_rate), min_temp]))

    return model

def train_model(input_params=None, x=None, y=None):

    if input_params is None:
        input_params = {}

    input_params = _parse_params(input_params)

    if K.backend() == 'theano':
        feature_axis = -3
    elif K.backend() == 'tensorflow':
        feature_axis = -1

    params = _default_params()
    params.update(input_params)
    model_function, model_params = _get_model(params['architecture'])
    params.update(model_params)
    params.update(input_params)
    params['image_dim_ordering'] = K.image_dim_ordering()

    model = model_function(**params)

    if params['initial_weights'] is not None:
        print('loading weights from: {}'.format(params['initial_weights']))
        model.load_weights(params['initial_weights'], by_name=params['load_weights_by_name'])

    output_folder = os.path.join(params['output_folder'], params['experiment_name']) + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if params['reporter_loss'] is None:
        loss = 'loss'
        val_loss = 'val_loss'

        losses = [loss, val_loss]
    else:
        loss = 'loss'
        val_loss = 'val_loss'
        reporter_loss = 'reporter_loss'
        val_reporter_loss = 'val_reporter_loss'
        losses = [loss, val_loss, reporter_loss, val_reporter_loss]

    plot_loss_callback = OutputHistory(output_folder + 'loss.png', output_folder + 'loss.json',
                                       save_graphs=params['save_graphs'], plot_logs=params['plot_logs'], metrics=losses)

    model_checkpoint = ModelCheckpoint(output_folder + 'weights.{epoch:02d}-{%s:.6f}.hdf5' % val_loss,
                                       save_best_only=params['save_best_only'],
                                       save_weights_only=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=True)

    json.dump(params, open(output_folder+'params.json', 'w'), sort_keys=True, indent=4)

    if params['flow_from_disk']:

        is_3d = params['conv_3d']
        is_recurrent = params['recurrent']
        if params['reporter_loss'] is not None:
            nb_targets = 2
        else:
            nb_targets = 1

        N = params['dataset_size']
        seq_len = params['tile_size'][-1]

        train_datagen = TwoImageIterator(params['data_location'],
                                        batch_size=params['batch_size'],
                                        is_3d=is_3d,
                                        is_recurrent=is_recurrent,
                                        target_size=tuple(params['tile_size'][:2]),
                                        nb_targets=nb_targets,
                                        N=N,
                                        shuffle=True,
                                        seq_len=seq_len)

        validation_datagen = TwoImageIterator(params['validation_location'],
                                              batch_size=params['batch_size'],
                                              is_3d=is_3d,
                                              is_recurrent=is_recurrent,
                                              target_size=tuple(params['tile_size'][:2]),
                                              nb_targets=nb_targets,
                                              seq_len=seq_len)

        if params['overfit']:
            validation_datagen = train_datagen

        if params['steps_per_epoch'] == -1:
            params['steps_per_epoch'] = train_datagen.N

        callbacks = [plot_loss_callback, model_checkpoint, lr_schedule]

        model.fit_generator(train_datagen,
                            steps_per_epoch=params['steps_per_epoch'],
                            epochs=params['nb_epoch'],
                            verbose=params["verbose"],
                            callbacks=callbacks,
                            validation_data=validation_datagen,
                            validation_steps=validation_datagen.N)
    else:
        if x is None or y is None:
            x, y = get_data_from_params(params)
            x, y = _preprocess_data(x, y, params)

        summary(params, x, y)

        if params['pad_target_features'] is not None:
            y = np.concatenate([y, ]*params['pad_target_features'], axis=feature_axis)

        if params['reporter_loss'] is not None:
            y = [y, y]

        model.fit(x, y, epochs=params['nb_epoch'], callbacks=[plot_loss_callback, model_checkpoint],
                  batch_size=params['batch_size'], verbose=params["verbose"], shuffle=True,
                  validation_split=params['validation_split'])

    return model


def train_model_crossvalidate(input_params, n_folds=4):

    if input_params is None:
        input_params = {}

    input_params = _parse_params(input_params)

    if K.backend() == 'theano':
        feature_axis = -3
    elif K.backend() == 'tensorflow':
        feature_axis = -1

    params = _default_params()
    params.update(input_params)
    model_function, model_params = _get_model(params['architecture'])
    params.update(model_params)
    params.update(input_params)
    params['image_dim_ordering'] = K.image_dim_ordering()

    model = []

    for i in range(n_folds):
        model.append(model_function(**params))

    if params['initial_weights'] is not None or params['cnn_weights'] is not None:
        print('loading weights from: {}'.format(params['initial_weights']))
        for i in range(n_folds):
            if params['initial_weights'] is not None:
                model[i].load_weights(params['initial_weights'], by_name=False)
            if params['cnn_weights'] is not None:
                model[i].load_weights(params['cnn_weights'], by_name=True)

    output_folder = os.path.join(params['output_folder'], params['experiment_name']) + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if params['reporter_loss'] is None:
        loss = 'loss'
        val_loss = 'val_loss'

        losses = [loss, val_loss]
    else:
        loss = 'loss'
        val_loss = 'val_loss'
        reporter_loss = 'reporter_loss'
        val_reporter_loss = 'val_reporter_loss'
        losses = [loss, val_loss, reporter_loss, val_reporter_loss]

    params['n_folds'] = n_folds

    json.dump(params, open(output_folder+'params.json', 'w'), sort_keys=True, indent=4)

    if params['flow_from_disk']:

        is_3d = params['recurrent'] or params['conv_3d']
        is_recurrent = params['recurrent']
        if params['reporter_loss'] is not None:
            nb_targets = 2
        else:
            nb_targets = 1

        N = params['dataset_size']
        seq_len = params['tile_size'][-1]

        crossvalidate_datagen = TwoImageCrossValidateIterator(params['data_location'], params['n_folds'],
                                    batch_size=params['batch_size'],
                                    is_3d=is_3d,
                                    is_recurrent=is_recurrent,
                                    target_size=tuple(params['tile_size'][:2]),
                                    nb_targets=nb_targets,
                                    N=N,
                                    shuffle=True,
                                    seq_len=seq_len)

        for i in range(n_folds):

            if not os.path.exists(output_folder + 'fold{}/'.format(i)):
                os.makedirs(output_folder + 'fold{}/'.format(i))

            crossvalidate_datagen.set_fold(i)

            train_datagen = crossvalidate_datagen.train_iterator()
            validation_datagen = crossvalidate_datagen.val_iterator()

            params['fold_{}_train'.format(i)] = train_datagen.filenames
            params['fold_{}_val'.format(i)] = validation_datagen.filenames
            json.dump(params, open(output_folder + 'fold{}/'.format(i) + 'params.json', 'w'), sort_keys=True, indent=4)

            plot_loss_callback = OutputHistory(output_folder + 'fold{}/'.format(i) + 'loss.png', output_folder + 'fold{}/'.format(i) + 'loss.json',
                                               save_graphs=params['save_graphs'], plot_logs=params['plot_logs'],
                                               metrics=losses)

            model_checkpoint = ModelCheckpoint(output_folder + 'fold{}/'.format(i) + 'weights.{epoch:02d}-{%s:.6f}.hdf5' % val_loss,
                                               save_best_only=params['save_best_only'],
                                               save_weights_only=True)
            lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=True)

            if params['steps_per_epoch'] == -1:
                params['steps_per_epoch'] = train_datagen.N

            callbacks = [plot_loss_callback, model_checkpoint, lr_schedule]

            model[i].fit_generator(train_datagen,
                                steps_per_epoch=params['steps_per_epoch'],
                                epochs=params['nb_epoch'],
                                verbose=params["verbose"],
                                callbacks=callbacks,
                                validation_data=validation_datagen,
                                validation_steps=validation_datagen.N)
    else:
        raise NotImplementedError

    return model

def summary(params, x, y):
    p = params.copy()

    print('experiment name: {}'.format(p.pop('experiment_name')))
    print('-'*20)
    print('training on {} image tiles'.format(x.shape[0]))
    print('fraction of blank tiles used: {}'.format(p.pop('blank_tiles_rate')))
    print('validation fraction: {}'.format(p.pop('validation_split')))
    print('batch_size: {}'.format(p.pop('batch_size')))
    print('-'*20)
    print('architecture: {}'.format(p.pop('architecture')))
    print('loss: {}'.format(p.pop('loss')))
    print('optimizer: {}'.format(p.pop('optimizer')))
    print('internal dropout rate: {}'.format(p.pop('internal_dropout_rate')))
    print('final dropout rate: {}'.format(p.pop('final_dropout_rate')))
    print('batch_norm: {}'.format(p.pop('batch_norm')))
    print('final activation: {}'.format(p.pop('final_activation')))
    print('-'*20)
    for pp in p:
        print('{}: {}'.format(pp, p[pp]))


def _preprocess_data(x, y, params, center=False):

    x /= 255.

    if center:
        for i in range(x.shape[0]):
            for j in range(x.shape[-1]):
                x[i,:,:,:,j] -= np.mean(x[i,:,:,:,j])
                x[i,:,:,:,j] /= np.std(x[i,:,:,:,j])

    if params['smoothing_method'] is not None:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i,j] = im_smooth(x[i,j], method=params['smoothing_method'], sigma=params['smoothing_sigma'])

    return x, y

def _make_training_generators(x, y, params):

        validation_split = params['validation_split']
        perm = np.random.permutation(x.shape[0])
        nb_train = int(x.shape[0]*(1-validation_split))
        x_train = x[perm[:nb_train]]
        x_valid = x[perm[nb_train:]]
        y_train = y[perm[:nb_train]]
        y_valid = y[perm[nb_train:]]

        train_datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True,
                                     elastic_deformations=True, shear_range=1., zoom_range=0.2, fill_mode='reflect')
        train_datagen.fit(x_train)
        train_datagen = train_datagen.flow(x_train, y_train, batch_size=params['batch_size'])

        return train_datagen, x_valid, y_valid



def load_data(params, slices='all', train_or_test='train'):
    im_train = []
    mask_train = []

    im_valid = []
    mask_valid = []

    if train_or_test == 'train':
        folder_list = params["training_cases"]
        perm = True
    elif train_or_test == 'test':
        folder_list = params["test_cases"]
        perm = False
    else:
        raise ValueError('Choose either \"train\" or \"test\"')


    data_dir = params["data_dir"]

    image_name = params["image_name"]
    label_name = params["label_name"]

    patch_size = params["patch_size"]
    stride_length = params["stride_length"]

    normalization = params["normalization"]
    debug = params["debug"]

    blank_tiles_rate = params["blank_tiles_rate"]
    validation_fraction = 0.05

    im_train = []

    for folder in folder_list:

        images = []
        for channel in range(params["num_channels"]):
            image_location = path.join(params["data_dir"], folder, params["image_name"][channel])
            nii_image = nib.load(image_location)
            image_data = nii_image.get_data().astype('float32')
            images.append(image_data)

        image = np.stack(images, axis=3)
        mask_image = nib.load(path.join(params["data_dir"], folder, params["label_name"])).get_data()
        mask_image = mask_image.reshape((mask_image.shape[0], mask_image.shape[1], mask_image.shape[2], 1))

        if debug:
            image = image[:256, :256, 10:13, :]
            mask_image = mask_image[:256, :256, 10:13, :]

        im_train.append(image4d_to_patches(image, patch_size, stride_length))
        mask_train.append(image4d_to_patches(mask_image, patch_size, stride_length))

    im_train = np.concatenate(im_train, axis=0)
    mask_train = np.concatenate(mask_train, axis=0)

    for channel in range(params["num_channels"]):
        mean = np.mean(im_train[:, channel, :, :])
        std = np.std(im_train[:, channel, :, :])

        if params["normalization"] == "whiten":
            offset = mean
            scale = std
            im_train[:, channel, :, :] -= offset
            im_train[:, channel, :, :] /= scale
        elif params["normalization"] == "scale":
            offset = 0.
            scale = 255.
            im_train[:, channel, :, :] -= offset
            im_train[:, channel, :, :] /= scale
        else:
            raise ValueError('Invalid normalization type')

    # --------------------
    if params['verbose']:
        print('X_max/min = ' + str(np.max(im_train)) + '/' + str(np.min(im_train)))
        print('Y_max/min = ' + str(np.max(mask_train)) + '/' + str(np.min(mask_train)))

    im_train = im_train.astype('float32')
    mask_train = mask_train.astype('float32')

    if blank_tiles_rate < 1:
        keep_list = []
        for i in range(im_train.shape[0]):
            if np.sum(mask_train[i, 0, :, :]) > 5 or rand() < blank_tiles_rate:
                keep_list.append(i)

        im_train = im_train[keep_list, :, :, :]
        mask_train = mask_train[keep_list, :, :, :]

    if params["smooth_input"]:
        for i in range(im_train.shape[0]):
            for c in range(im_train.shape[1]):
                mv = np.max(im_train[i,c,:,:])
                # im_train[i,c,:,:] = median(im_train[i,c,:,:], disk(3))*(float(mv)/255)
                im_train[i,c,:,:] = gaussian(im_train[i,c,:,:].astype(float)/mv, sigma=1)*mv

    num_valid = round(im_train.shape[0] * validation_fraction)
    num_valid = max(num_valid, 1)

    perm = permutation(im_train.shape[0])
    im_valid = im_train[perm[:num_valid], :, :, :]
    mask_valid = mask_train[perm[:num_valid], :, :, :]

    im_train = im_train[perm[num_valid:], :, :, :]
    mask_train = mask_train[perm[num_valid:], :, :, :]

    return (im_train, mask_train), (im_valid, mask_valid)


def load_data_3d(params, slices='all', train_or_test='train'):

    if train_or_test == 'train':
        folder_list = params["training_cases"]
        perm = True
    elif train_or_test == 'test':
        folder_list = params["test_cases"]
        perm = False
    else:
        raise ValueError('Choose either \"train\" or \"test\"')


    data_dir = params["data_dir"]

    image_name = params["image_name"]
    label_name = params["label_name"]

    patch_size = params["patch_size"]
    stride_length = params["stride_length"]

    normalization = params["normalization"]
    debug = params["debug"]

    blank_tiles_rate = params["blank_tiles_rate"]
    validation_fraction = 0.1

    im_train = []
    mask_train = []

    for folder in folder_list:

        images = []
        for channel in range(params["num_channels"]):
            image_location = path.join(params["data_dir"], folder, params["image_name"][channel])
            nii_image = nib.load(image_location)
            image_data = nii_image.get_data().astype('float32')
            images.append(image_data)

        image = np.stack(images, axis=3)
        mask_image = nib.load(path.join(params["data_dir"], folder, params["label_name"])).get_data()
        mask_image = mask_image.reshape((mask_image.shape[0], mask_image.shape[1], mask_image.shape[2], 1))

        if len(image.shape) > 4:
            image = image.reshape((image.shape[0],image.shape[1],image.shape[2],image.shape[3]))

        if debug:
            image = image[:256, :256, 10:13, :]
            mask_image = mask_image[:256, :256, 10:13, :]

        # Stacks upper and lower slice in the channel dimension
        image_up = image[:,:,0:-2,:]
        image_down = image[:,:,2:,:]
        image = np.concatenate((image[:,:,1:-1,:],image_up, image_down), axis=3)

        mask_image = mask_image[:,:,1:-1]

        im_train.append(image4d_to_patches(image, patch_size, stride_length))
        mask_train.append(image4d_to_patches(mask_image, patch_size, stride_length))

    im_train = np.concatenate(im_train, axis=0)
    mask_train = np.concatenate(mask_train, axis=0)

    for channel in range(im_train.shape[1]):
        mean = np.mean(im_train[:, channel, :, :])
        std = np.std(im_train[:, channel, :, :])

        if params["normalization"] == "whiten":
            offset = mean
            scale = std
            im_train[:, channel, :, :] -= offset
            im_train[:, channel, :, :] /= scale
        elif params["normalization"] == "scale":
            offset = 0.
            scale = 255.
            im_train[:, channel, :, :] -= offset
            im_train[:, channel, :, :] /= scale
        else:
            raise ValueError('Invalid normalization type')

    # --------------------
    if params['verbose']:
        print('X_max/min = ' + str(np.max(im_train)) + '/' + str(np.min(im_train)))
        print('Y_max/min = ' + str(np.max(mask_train)) + '/' + str(np.min(mask_train)))

    im_train = im_train.astype('float32')
    mask_train = mask_train.astype('float32')

    if blank_tiles_rate < 1:
        keep_list = []
        for i in range(im_train.shape[0]):
            if np.sum(mask_train[i, 0, :, :]) > 5 or rand() < blank_tiles_rate or np.sum(im_train[i, 1, :, :] > 150) > 100:
                keep_list.append(i)

        im_train = im_train[keep_list, :, :, :]
        mask_train = mask_train[keep_list, :, :, :]

    if params["smooth_input"]:
        for i in range(im_train.shape[0]):
            for c in range(im_train.shape[1]):
                mv = np.max(im_train[i,c,:,:])
                im_train[i,c,:,:] = gaussian(im_train[i,c,:,:].astype(float)/mv, sigma=2)*mv

    num_valid = round(im_train.shape[0] * validation_fraction)
    num_valid = max(num_valid, 1)

    perm = permutation(im_train.shape[0])
    im_valid = im_train[perm[:num_valid], :, :, :]
    mask_valid = mask_train[perm[:num_valid], :, :, :]

    im_train = im_train[perm[num_valid:], :, :, :]
    mask_train = mask_train[perm[num_valid:], :, :, :]

    return (im_train, mask_train), (im_valid, mask_valid)


def load_data_lstm(params):

    folder_list = params["training_cases"]

    patch_size = params["patch_size"]
    stride_length = params["stride_length"]
    debug = params["debug"]

    blank_tiles_rate = params["blank_tiles_rate"]

    im_train = []
    mask_train = []

    for folder in folder_list:

        images = []
        for channel in range(params["num_channels"]):
            image_location = path.join(params["data_dir"], folder, params["image_name"][channel])
            nii_image = nib.load(image_location)
            image_data = nii_image.get_data().astype('float32')
            images.append(image_data)

        image = np.stack(images, axis=3)
        mask_image = nib.load(path.join(params["data_dir"], folder, params["label_name"])).get_data()
        mask_image = mask_image.reshape((mask_image.shape[0], mask_image.shape[1], mask_image.shape[2], 1))

        if len(image.shape) > 4:
            image = image.reshape((image.shape[0], image.shape[1], image.shape[2], image.shape[3]))

        if debug:
            image = image[:256, :256, 10:13, :]
            mask_image = mask_image[:256, :256, 10:13, :]

        # Stacks upper and lower slice in the channel dimension

        mask_image = mask_image[:, :, 1:-1]

        im_train.append(image4d_to_stack_patches(image, patch_size, stride_length))
        mask_train.append(image4d_to_stack_patches(mask_image, patch_size, stride_length))

    im_train = np.concatenate(im_train, axis=0)
    mask_train = np.concatenate(mask_train, axis=0)

    for channel in range(im_train.shape[1]):
        mean = np.mean(im_train[:, channel, :, :])
        std = np.std(im_train[:, channel, :, :])

        if params["normalization"] == "whiten":
            offset = mean
            scale = std
            im_train[:, channel, :, :] -= offset
            im_train[:, channel, :, :] /= scale
        elif params["normalization"] == "scale":
            offset = 0.
            scale = 255.
            im_train[:, channel, :, :] -= offset
            im_train[:, channel, :, :] /= scale
        else:
            raise ValueError('Invalid normalization type')

    # --------------------
    if params['verbose']:
        print('X_max/min = ' + str(np.max(im_train)) + '/' + str(np.min(im_train)))
        print('Y_max/min = ' + str(np.max(mask_train)) + '/' + str(np.min(mask_train)))

    im_train = im_train.astype('float32')
    mask_train = mask_train.astype('float32')

    if blank_tiles_rate < 1:
        keep_list = []
        for i in range(im_train.shape[0]):
            if np.sum(mask_train[i, 0, :, :]) > 5 or rand() < blank_tiles_rate or np.sum(im_train[i, 1, :, :] > 150) > 100:
                keep_list.append(i)

        im_train = im_train[keep_list, :, :, :]
        mask_train = mask_train[keep_list, :, :, :]

    if params["smooth_input"]:
        for i in range(im_train.shape[0]):
            for c in range(im_train.shape[1]):
                mv = np.max(im_train[i, c, :, :])
                # im_train[i,c,:,:] = median(im_train[i,c,:,:], disk(3))*(float(mv)/255)
                im_train[i, c, :, :] = gaussian(im_train[i, c, :, :].astype(float) / mv, sigma=2) * mv

    num_valid = round(im_train.shape[0] * validation_fraction)
    num_valid = max(num_valid, 1)

    perm = permutation(im_train.shape[0])
    im_valid = im_train[perm[:num_valid], :, :, :]
    mask_valid = mask_train[perm[:num_valid], :, :, :]

    im_train = im_train[perm[num_valid:], :, :, :]
    mask_train = mask_train[perm[num_valid:], :, :, :]

    return (im_train, mask_train), (im_valid, mask_valid)

def train_and_predict(params):

    model_location = params["model"]

    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    if params["use_3d"]:
        (im_train, mask_train), (im_valid, mask_valid) = load_data_3d(params)
        params["num_channels"] *= 3
    elif params['architecture'] == 'conv_lstm':
        (im_train, mask_train), (im_valid, mask_valid) = load_data_lstm(params)
    else:
        (im_train, mask_train), (im_valid, mask_valid) = load_data(params)

    if params["flat_output"]:
        mask_train = mask_train.reshape((mask_train.shape[0], -1))
        mask_valid = mask_valid.reshape((mask_valid.shape[0], -1))

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = _get_network(params)

    if params['initial_weights'] is not False:
        model.load_weights(params['initial_weights'])
        # model = load_initial_weights(model, params)

    if not os.path.exists(model_location):
        os.mkdir(model_location)

    model_location = path.join(model_location, 'weights.{epoch:02d}-{val_loss:.6f}.hdf5')
    model_checkpoint = ModelCheckpoint(model_location, monitor='val_loss', save_best_only=False, verbose=1)
    early_stopping = EarlyStopping(verbose=1, patience=3)
    print_cb = epoch_print_callback()

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    print('data max: ' + str(np.max(im_train)))
    print('data min: ' + str(np.min(im_train)))
    print('mask max: ' + str(np.max(mask_train)))
    print('mask min: ' + str(np.min(mask_train)))
    if params['augment_data']:
        data_gen_args = dict(rotation_range=90.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             vertical_flip=True,
                             horizontal_flip=True,
                             zoom_range=0.2,
                             fill_mode='reflect',
                             elastic_deformations=True)
        image_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = rand()

        print(image_datagen.dim_ordering)
        image_datagen.fit(im_train)

        image_flow = image_datagen.flow(im_train, mask_train, batch_size=batch_size)

        model.fit_generator(image_flow, samples_per_epoch=im_train.shape[0], nb_epoch=num_epochs, verbose=params["verbose"],
                  callbacks=[model_checkpoint], validation_data=(im_valid, mask_valid))

    else:
        model.fit(im_train, mask_train, batch_size=batch_size, nb_epoch=num_epochs, verbose=params["verbose"], shuffle=True,
                  callbacks=[model_checkpoint], validation_data=(im_valid, mask_valid))

    pred = model.predict(im_train)
    print('predicted max: ' + str(np.max(pred)))
    print('predicted min: ' + str(np.min(pred)))

    print('-' * 30)
    print('Completed fitting model...')
    print('-' * 30)

def load_initial_weights(network, params):


    dummy_network = _get_network(params)
    dummy_network.load_weights(params['initial_weights'])

    w_3d = network.layers[1].get_weights()
    w_2d = dummy_network.layers[1].get_weights()

    # w_3d[0] = np.zeros(w_3d[0].shape)
    s = w_2d[0].shape[1]
    w_3d[0][:, :s, :, :] = w_2d[0][:, :, :, :]
    w_3d[1] = w_2d[1]
    network.layers[1].set_weights(w_3d)

    for layer1, layer2 in zip(network.layers[2:], dummy_network.layers[2:]):
        w_3d = layer1.get_weights()
        w_2d = layer2.get_weights()
        for i in range(len(w_2d)):
            w_3d[i] = w_2d[i]

        layer1.set_weights(w_3d)

    network.compile(optimizer='adam', loss='binary_crossentropy')

    return network


def pretrain_load_data(params, case):
    """

    - Loads the data
    - Randomly splits into 90% training and 10% test.
    - Validation set is completely separate.

    :param c1:
    :return:
    """

    fname = params['data_dir'] + 'ves_' + 'Train' + '_' + params['cases'][case] + "_" + str(
        params['image_size']) + '_norm' + str(params['norm']) + '.npz'
    fnpz = np.load(fname)

    X = fnpz['X']
    Y = fnpz['Y']
    X = np.reshape(X, (-1, 1, params['image_size'], params['image_size']))
    X = X.astype('float32')
    Y = np.reshape(Y, (-1, 1, params['image_size'], params['image_size']))
    Y = Y.astype('float32')

    X = X / X.max()
    Y = Y / Y.max()

    # Double the amount of data by swapping axes
    # X = np.concatenate((X, np.swapaxes(X, 2, 3)), axis=0)
    # Y = np.concatenate((Y, np.swapaxes(Y, 2, 3)), axis=0)

    # Random permutation
    ind1 = np.random.permutation(X.shape[0])
    X = X[ind1, :]
    Y = Y[ind1, :]

    print('X type and shape:', X.dtype, X.shape)
    print('X.min():', X.min())
    print('X.max():', X.max())

    return X, Y

def pretrain_unet(params):

    model = _get_network(params)
    model_location = params['model']
    os.mkdir(model_location)

    for e in range(params['num_epochs']):
        # print("Epoch {}".format(e), file=sys.stderr)
        ll = []

        # Looping over the dataset of each case separately
        for c1 in range(len(params['training_cases'])):
            print("Case: ", params['training_cases'][c1])

            # Load data for each folder
            fname = params['training_cases'][c1] + params['image_name'][0]
            img_path = params['data_dir'] + fname

            svstr = img_path + '.npz'
            fnpz = np.load(svstr)

            if params['num_channels'] == 2:
                X = fnpz['X']
            elif params['num_channels'] == 3:
                X1 = fnpz['X1']
                X2 = fnpz['X2']
                X = np.stack((X1,X2),axis=1)

            X = X.astype('float32')
            Y = fnpz['Y']
            Y = Y.astype('float32')

            X = X.reshape((-1, params['num_channels'], params['patch_size'], params['patch_size']))

            if params["flat_output"]:
                Y_out = Y.reshape((Y.shape[0], -1))
            else:
                Y_out = Y.reshape((-1, 1, params['patch_size'], params['patch_size']))

            if params['normalization'] == 'scale':
                X /= 255.
                Y_out /= 255.

            print('X_min: ' + str(np.min(X)) + ' X_max: ' + str(np.max(X)), flush=True )
            print('Y_min: ' + str(np.min(Y_out)) + ' Y_max: ' + str(np.max(Y_out)), flush=True)

            model_name = path.join(model_location, 'weights.'+str(e)+'-{val_loss:.6f}.hdf5')
            model_checkpoint = ModelCheckpoint(model_name, monitor='loss', save_best_only=False, verbose=1)
            early_stopping = EarlyStopping(verbose=1, patience=3)		
		
            if params['augment_data']:
                datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True, elastic_deformations=True)
                datagen.fit(X)
                loss = model.fit_generator(datagen.flow(X, Y_out, batch_size=params['batch_size']), samples_per_epoch=len(X),
                                           nb_epoch=1, verbose=params['verbose'], callbacks=[model_checkpoint], validation_split=0.1)
            else:
                loss = model.fit(X, Y_out, batch_size=params['batch_size'], nb_epoch=1, verbose=params["verbose"],
                          callbacks=[model_checkpoint], validation_split=0.1)

            ll.append(loss.history['loss'][0])

            print("loss {0}: {1:.4f}. ".format(c1, loss.history['loss'][0]), end="", file=sys.stderr, flush=True)

        print("", file=sys.stderr)
        print("Mean loss: {0:.4f}".format(np.mean(ll)), file=sys.stderr, flush=True)

    model.save_weights(params['model'])




'''Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
Modified by He Xie 08/2016

For image segmentation problem data augmentation.
Transform train img data and mask img data simultaneously and in the same fashion.
Omit flow from directory function.
'''


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, dim_ordering='default', scale=True):
    from PIL import Image
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])


def img_to_array(img, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.
    Assume X is train img, Y is train label (same size as X with only 0 and 255 for values)
    # Arguments
        featurewise_center: set input mean to 0 over the dataset. Only to X
        samplewise_center: set each sample mean to 0. Only to X
        featurewise_std_normalization: divide inputs by std of the dataset. Only to X
        samplewise_std_normalization: divide each input by its std. Only to X
        zca_whitening: apply ZCA whitening. Only to X
        rotation_range: degrees (0 to 180). To X and Y
        width_shift_range: fraction of total width. To X and Y
        height_shift_range: fraction of total height. To X and Y
        shear_range: shear intensity (shear angle in radians). To X and Y
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range. To X and Y
        channel_shift_range: shift range for each channels. Only to X
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'. For Y, always fill with constant 0
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally. To X and Y
        vertical_flip: whether to randomly flip images vertically. To X and Y
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation). Only to X
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 elastic_deformations=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 dim_ordering='default',
                 volume=False):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering

        if not volume:
            if dim_ordering == 'th':
                self.channel_index = 1
                self.row_index = 2
                self.col_index = 3
            if dim_ordering == 'tf':
                self.channel_index = 3
                self.row_index = 1
                self.col_index = 2
        else:
            if dim_ordering == 'th':
                self.channel_index = 2
                self.row_index = 3
                self.col_index = 4
            if dim_ordering == 'tf':
                self.channel_index = 4
                self.row_index = 2
                self.col_index = 3

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def standardize(self, x):
        # Only applied to X
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        return x

    def random_transform(self, x, y):
        # Need to modify to transform both X and Y ---- to do
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        if self.elastic_deformations:
            x, y = elastic_distortion(image=x, mask=y, alpha=50, sigma=6, channel_index=img_channel_index)

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

        if len(x.shape) > 3:
            for i in range(x.shape[0]):
                x[i] = apply_transform(x[i], transform_matrix, img_channel_index-1,
                                    fill_mode=self.fill_mode, cval=self.cval)
                # For y, mask data, fill mode constant, cval = 0
                y[i] = apply_transform(y[i], transform_matrix, img_channel_index-1,
                                    fill_mode=self.fill_mode, cval=self.cval)
        else:
            x = apply_transform(x, transform_matrix, img_channel_index,
                                   fill_mode=self.fill_mode, cval=self.cval)
            # For y, mask data, fill mode constant, cval = 0
            y = apply_transform(y, transform_matrix, img_channel_index,
                                   fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Only applied to X
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        # ?
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        batch_y = np.zeros(tuple([current_batch_size] + list(self.y.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            label = self.y[j]
            x, label = self.image_data_generator.random_transform(x.astype('float32'), label.astype("float32"))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = label
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
                mask = array_to_img(batch_y[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}_mask.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                mask.save(os.path.join(self.save_to_dir, fname))
        return batch_x, batch_y


# See http://arxiv.org/pdf/1003.0358.pdf for the description of the method
def elastic_distortion(image, mask=None, kernel_dim=31, sigma=6, alpha=47, channel_index=2):

    # Returns gaussian kernel in two dimensions
    # d is the square kernel edge size, it must be an odd number.
    # i.e. kernel is of the size (d,d)
    image_input_shape = image.shape

    def gaussian_kernel(d, sigma):
        if d % 2 == 0:
            raise ValueError("Kernel edge size must be an odd number")

        cols_identifier = np.int32(
            np.ones((d, d)) * np.array(np.arange(d)))
        rows_identifier = np.int32(
            np.ones((d, d)) * np.array(np.arange(d)).reshape(d, 1))

        kernel = np.exp(-1. * ((rows_identifier - d/2)**2 +
            (cols_identifier - d/2)**2) / (2. * sigma**2))
        kernel *= 1. / (2. * math.pi * sigma**2)  # normalize
        return kernel

    if 0 <= channel_index < len(image.shape):
        imshape = image.shape[:channel_index] + image.shape[(channel_index+1):]
        # imshape = imshape[-2:]
    else:
        raise ValueError('invalid channel index')

    field_x = np.random.uniform(low=-1, high=1, size=imshape[-2:]) * alpha
    field_y = np.random.uniform(low=-1, high=1, size=imshape[-2:]) * alpha

    kernel = gaussian_kernel(kernel_dim, sigma)

    # Distortion fields convolved with the gaussian kernel
    # This smoothes the field out.
    field_x = convolve2d(field_x, kernel, mode="same")
    field_y = convolve2d(field_y, kernel, mode="same")

    d = field_x.shape[0]

    cols_identifier = np.int32(np.ones((d, d))*np.array(np.arange(d)))
    rows_identifier = np.int32(
        np.ones((d, d))*np.array(np.arange(d)).reshape(d, 1))

    down_row = np.int32(np.floor(field_x)) + rows_identifier
    top_row = np.int32(np.ceil(field_x)) + rows_identifier
    down_col = np.int32(np.floor(field_y)) + cols_identifier
    top_col = np.int32(np.ceil(field_y)) + cols_identifier

    x1 = down_row.flatten()
    y1 = down_col.flatten()
    x2 = top_row.flatten()
    y2 = top_col.flatten()

    if len(image.shape) < 3:
        distorted_image = apply_warp(image, x1, x2, y1, y2, rows_identifier, cols_identifier, field_x, field_y)
    if len(image.shape) == 3:
        chans = []
        for chan in range(image.shape[channel_index]):
            if 0 <= channel_index < len(image.shape):
                channel = np.rollaxis(image, channel_index)[chan]
            else:
                raise ValueError('invalid channel index')

            chans.append(apply_warp(channel, x1, x2, y1, y2, rows_identifier, cols_identifier, field_x, field_y))
        distorted_image = np.stack(chans, axis=channel_index)

    if len(image.shape) == 4:
        distorted_image = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[3]):
                channel = image[i,:,:,j].reshape(imshape[-2:])
                distorted_image[i,:,:,j] = apply_warp(channel, x1, x2, y1, y2, rows_identifier, cols_identifier, field_x, field_y)


    distorted_image = distorted_image.reshape(image_input_shape)

    if mask is not None:
        mask_input_shape = mask.shape

        if len(mask_input_shape)==3:
            distorted_mask = apply_warp(mask.reshape((d,d)), x1, x2, y1, y2, rows_identifier, cols_identifier, field_x, field_y)
            distorted_mask = np.reshape(distorted_mask, mask_input_shape)

        if len(mask_input_shape)==4:
            distorted_mask = np.zeros(mask_input_shape)
            for i in range(mask_input_shape[0]):
                channel = mask[i].reshape((d,d))
                distorted_mask[i,:,:,0] = apply_warp(channel, x1, x2, y1, y2, rows_identifier, cols_identifier, field_x, field_y)

        return distorted_image, distorted_mask
    else:
        return distorted_image


def apply_warp(image, x1, x2, y1, y2, rows_identifier, cols_identifier, field_x, field_y):

    d = image.shape[0]
    padded_image = np.pad(
        image, pad_width=d, mode="reflect")

    Q11 = padded_image[d + x1, d + y1]
    Q12 = padded_image[d + x1, d + y2]
    Q21 = padded_image[d + x2, d + y1]
    Q22 = padded_image[d + x2, d + y2]
    x = (rows_identifier + field_x).flatten()
    y = (cols_identifier + field_y).flatten()

    # Bilinear interpolation algorithm is as described here:
    # https://en.wikipedia.org/wiki/Bilinear_interpolation#Algorithm
    distorted_image = (1. / ((x2 - x1) * (y2 - y1))) * (
        Q11 * (x2 - x) * (y2 - y) +
        Q21 * (x - x1) * (y2 - y) +
        Q12 * (x2 - x) * (y - y1) +
        Q22 * (x - x1) * (y - y1))

    distorted_image = distorted_image.reshape((d, d))
    return distorted_image


if __name__ == '__main__':
    train_and_predict()
