import json
from pickle import load, dump
import os
import yaml
from functools import reduce

from unet_core.networks.convolutional_networks import *
from unet_core.networks.recurrent_networks import *
from unet_core.utils.data_utils import reconstruct_slice, reconstruct_feature
from unet_core.utils.maths import sigmoid
from unet_core.utils.stack_utils import take_strided_stacks_from_single_image
from keras.utils.conv_utils import convert_kernel


class KerasModel(object):

    def __init__(self, input_params=None, ignore_blank_tiles=True, save_debug_images=False, **kwargs):

        if type(input_params) is str:
            with open(input_params) as f:
                self.params = yaml.load(f)
        else:
            self.params = input_params

        params = _default_params()
        params.update(self.params)
        self.model_function, model_params = _get_model(params['architecture'])
        params.update(model_params)
        params.update(self.params)

        self.params = params

        self.saved_debug_im = False
        self.save_debug_images = save_debug_images
        self.auto_calibrated = False
        self.tile_size = self.params['tile_size']
        self.stride_length = self.params['stride_length']
        self.blank_tile_value = self.params['blank_tile_value']
        self.num_channels = len(self.params['channels'])
        self.ignore_blank_tiles = ignore_blank_tiles
        self.output_feature = params['output_feature']
        self.is_logit = params['is_logit']

        if 'use_3d' in self.params.keys():
            self.use_3d = self.params['use_3d']
        else:
            self.use_3d = False

        if self.use_3d:
            self.num_channels *= 3

        if self.tile_size != 'full':
            self.network = self.model_function(**self.params)
            if self.params['initial_weights'] is not None:
                self.load_weights(self.params['initial_weights'], by_name=False)

    def normalize(self, data):

        if self.params["data_normalization"] == "whiten":
            for i in range(data.shape[1]):
                offset = np.mean(data[:, i, :, :])
                scale = np.std(data[:, i, :, :])
                data[:, i, :, :] -= offset
                data[:, i, :, :] /= scale
        elif self.params["data_normalization"] == "scale":
            offset = 0.
            scale = 255.
            data -= offset
            data /= scale
        elif self.params["data_normalization"] == "none" or self.params["data_normalization"] is None:
            pass
        else:
            raise ValueError('Invalid normalization type')

        return data

    def segment_slice(self, slice, output_feature=0, is_logit=False):

        if self.auto_calibrated is False:
            self.calibrate(slice.shape)

        if isinstance(output_feature, list):
            nb_output_features = len(output_feature)
        else:
            nb_output_features = 1

        rows, cols = slice.shape[0], slice.shape[1]
        tile_image = True

        slice = np.reshape(slice, (rows, cols, 1, self.num_channels))

        if tile_image:
            image_tiles = take_strided_stacks_from_single_image(slice, self.stride_length + [1,], self.tile_size[0], self.tile_size[1], seq_length=1)
        else:
            image_tiles = slice.reshape((1, self.num_channels, rows, cols))

        image_tiles = self.normalize(image_tiles)

        if self.ignore_blank_tiles:
            keep_list, throw_list = self._check_for_blanks(image_tiles)
        else:
            keep_list = range(image_tiles.shape[0])

        predicted = self.blank_tile_value*np.ones((image_tiles.shape[0], nb_output_features, self.tile_size[0], self.tile_size[1]))

        if len(keep_list) > 0:
            sub_stack = image_tiles[keep_list, :, :, :]

            p = self.network.predict(sub_stack, verbose=1, batch_size=self.params['batch_size'])

            if K.image_dim_ordering() == 'tf':
                p = p.transpose((0, 3, 1, 2))

            for i,j in enumerate(keep_list):
                predicted[j, :, :, :] = p[i,output_feature,:,:]

        if tile_image:
            reconstructed_slice = reconstruct_slice(tiles=predicted, image_dims=(rows, cols),
                                                    tile_size=(self.tile_size[0], self.tile_size[1]),
                                                    stride_length=self.stride_length)
        else:
            reconstructed_slice = predicted

        if self.save_debug_images:
            plt.imsave("debugslice_output.png",255*reconstructed_slice.reshape((rows,cols)))
            plt.imsave("debugslice_endo.png",255*slice[:,:,0,0])
            plt.imsave("debugslice_tumour.png",255*slice[:,:,0,1])

        if is_logit:
            return 255*sigmoid(reconstructed_slice)
        else:
            return 255*reconstructed_slice

    def calibrate(self, slice_shape):
        tt = np.zeros(2, dtype='uint16')
        ss = np.zeros(2, dtype='uint16')
        recompile = False

        if self.tile_size == 'full':
            self.tile_size = slice_shape[:2]
            recompile = True

        for i in range(2):
            if float(slice_shape[i])/float(self.tile_size[i]) % 1 == 0:
                tt[i] = self.tile_size[i]
                ss[i] = self.stride_length[i]
            else:
                stride_ratio = self.tile_size[i]/self.stride_length[i]
                ratio = float(slice_shape[i])/float(self.tile_size[i])
                factor_list = self.factors(slice_shape[i])
                factor_list = [x for x in factor_list if x < self.tile_size[i]]
                dist = (ratio-slice_shape[i]/np.array(factor_list))**2

                new_ratio = factor_list[np.argmin(dist)]
                tt[i] = new_ratio
                ss[i] = new_ratio/stride_ratio
                recompile = True

        self.tile_size[:2] = [tt[0], tt[1]]
        self.stride_length[:2] = [ss[0], ss[1]]
        self.auto_calibrated = True
        self.params['tile_size'][:2] = self.tile_size[:2]
        self.params['stride_length'][:2] = self.stride_length[:2]

        print('Stack dimensions: {}'.format(slice_shape))
        print('Calibrated tile size: {}'.format(self.tile_size))
        print('Calibrated stride length: {}'.format(self.stride_length))

        if recompile:
            self.network = self.model_function(**self.params)
            if self.params['initial_weights'] is not None:
                self.load_weights(self.params['initial_weights'], by_name=self.params['load_weights_by_name'])

    def load_weights(self, weights_path, by_name=False):



        print('loading weights from: {}'.format(weights_path))
        # model_pre = self.model_function(**self.params)
        # self.network.save_weights('tmp_weights.hdf5')
        # model_pre.load_weights('tmp_weights.hdf5')
        # os.remove('tmp_weights.hdf5')
        if self.params['image_dim_ordering'] == K.image_dim_ordering():
            self.network.load_weights(weights_path, by_name=by_name)
        elif self.params['image_dim_ordering'] == 'th' and K.image_dim_ordering() == 'tf':
            self.load_th_weights_for_tf(weights_path, by_name=by_name)
        # model_post = self.network
        # changed = {True: 'changed', False: 'unchanged'}
        #
        # for layer_pre, layer_post in zip(model_pre.layers, model_post.layers):
        #     layer_changed = False
        #     for w_pre, w_post in zip(layer_pre.get_weights(), layer_post.get_weights()):
        #         if not np.array_equal(w_pre, w_post):
        #             layer_changed = True
        #
        #     print("{}: {}".format(layer_post.name, changed[layer_changed]))

    def load_th_weights_for_tf(self, weights_path, model=None, by_name=False):
        image_dim_ordering = K.image_dim_ordering()
        image_data_format = K.image_data_format()

        if model is None:
            model_tf = self.network
        else:
            model_tf = model

        if not isinstance(weights_path, list):
            weights_path = [weights_path, ]

        for weight in weights_path:
            K.set_image_dim_ordering('th')
            K.set_image_data_format('channels_first')
            model_th = self.model_function(**self.params)
            model_th.load_weights(weight)
            K.set_image_dim_ordering(image_dim_ordering)
            K.set_image_data_format(image_data_format)

            for layer_th, layer_tf in zip(model_th.layers, model_tf.layers):
                if any(x in layer_th.__class__.__name__ for x in ['conv', 'Conv', 'permute', 'Permute']):
                    weights_th = layer_th.get_weights()

                    if weights_th:
                        kernel = weights_th[0]
                        bias = weights_th[1]

                        converted_kernel = convert_kernel(kernel)

                        weights_tensorflow = [converted_kernel, bias]

                        layer_tf.set_weights(weights_tensorflow)
                elif 'TimeDistributed' in layer_th.__class__.__name__ and any(x in layer_th.layer.__class__.__name__ for x in ['conv', 'Conv', 'permute', 'Permute']):
                    weights_th = layer_th.get_weights()

                    if weights_th:
                        kernel = weights_th[0]
                        bias = weights_th[1]

                        converted_kernel = convert_kernel(kernel)

                        weights_tensorflow = [converted_kernel, bias]

                        layer_tf.set_weights(weights_tensorflow)
                else:
                    layer_tf.set_weights(layer_th.get_weights())


    @staticmethod
    def factors(n):
        return np.array(reduce(list.__add__, ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    @staticmethod
    def _check_for_blanks(image_list, threshold =0.8, rate=5e-4):

        """ignore image if values in specified channel do not exceed threshold at given rate"""

        channel_to_check = 0

        keep_list = []
        throw_list = []

        if image_list.ndim == 4 and K.image_dim_ordering() == 'th':
            size = image_list.shape[2]*image_list.shape[3]
            amount = rate*size
            amount = min(amount, 1000)
            for i in range(image_list.shape[0]):
                slice = image_list[i,channel_to_check]
                if np.sum(slice > threshold) > amount:
                    keep_list.append(i)
                else:
                    throw_list.append(i)

        if image_list.ndim == 4 and K.image_dim_ordering() == 'tf':
            size = image_list.shape[-3] * image_list.shape[-2]
            amount = rate * size
            amount = min(amount, 1000)
            for i in range(image_list.shape[0]):
                slice = image_list[i, :, :, channel_to_check]
                if np.sum(slice > threshold) > amount:
                    keep_list.append(i)
                else:
                    throw_list.append(i)

        if image_list.ndim == 5 and K.image_dim_ordering() == 'th':
            size = image_list.shape[-2]*image_list.shape[-1]
            amount = rate*size
            amount = min(amount, 1000)
            for i in range(image_list.shape[1]):
                slice = image_list[0, i, channel_to_check]
                if np.sum(slice > threshold) > amount:
                    keep_list.append(i)
                else:
                    throw_list.append(i)

        if image_list.ndim == 5 and K.image_dim_ordering() == 'tf':
            size = image_list.shape[-3] * image_list.shape[-2]
            amount = rate * size
            amount = min(amount, 1000)
            for i in range(image_list.shape[1]):
                slice = image_list[:, i, :, :, channel_to_check]
                if np.sum(slice > threshold) > amount:
                    keep_list.append(i)
                else:
                    throw_list.append(i)

        return keep_list, throw_list


    @staticmethod
    def default_params(params=None):

        if params is None:
            params = {}

        def_params = {}
        def_params['tile_size'] = (512, 512)
        def_params['stride_length'] = (512, 512)
        def_params['image_name'] = None
        def_params['label_name'] = None
        def_params['model'] = None

        # TRAINING PARAMETERS
        def_params['normalization'] = "scale"
        def_params['init_style'] = 'glorot_uniform'
        def_params['training_phase'] = "supervised"
        def_params['loss'] = 'mean_squared_error'
        def_params['optimizer'] = None
        def_params['flat_output'] = False
        def_params['dropout_rate'] = 0
        def_params['learning_rate'] = 0

        # NETWORK PARAMETERS
        def_params['conv_size'] = 3

        def_params.update(params)
        params = def_params

        return params


class LSTMModel(KerasModel):

    def __init__(self, params,
                 spatial_pad_size=3, temporal_pad_size=3, **kwargs):

        super(LSTMModel, self).__init__(params, **kwargs)

        self.unpadded_tile_size = list(self.params['tile_size'])
        self.spatial_pad_size = spatial_pad_size
        self.temporal_pad_size = temporal_pad_size

        self.tile_size[0] += 2*self.spatial_pad_size
        self.tile_size[1] += 2*self.spatial_pad_size

        self.auto_calibrated = False
        self.saved_debug_im = False
        self.slice_batch_size = params['tile_size'][2]

        if self.spatial_pad_size == 0:
            self.s_pad_l = None
            self.s_pad_u = None
        else:
            self.s_pad_l = self.spatial_pad_size
            self.s_lad_u = -self.spatial_pad_size

        if self.temporal_pad_size == 0:
            self.t_pad_l = None
            self.t_pad_u = None
        else:
            self.t_pad_l = self.temporal_pad_size
            self.t_pad_u = -self.temporal_pad_size

    def calibrate(self, slice_shape):
        super(LSTMModel, self).calibrate(slice_shape)
        self.unpadded_tile_size = list(self.params['tile_size'])
        self.tile_size[0] += 2*self.spatial_pad_size
        self.tile_size[1] += 2*self.spatial_pad_size


    def segment_stack(self, stack, verbose=False):

        if len(stack.shape) == 3:
            rows, cols, nb_slices = stack.shape
            nb_channels = 1
        if len(stack.shape) == 4:
            rows, cols, nb_slices, nb_channels = stack.shape

        nb_output_features = 1
        # TODO: Implement multiple output channels

        tile_image = True
        if len(stack.shape) == 3:
            stack = stack[:, :, :, np.newaxis]

        slice = np.reshape(stack, (rows, cols, nb_slices, nb_channels))

        if tile_image:
            strides = (self.params['stride_length'][0], self.params['stride_length'][0], nb_slices)
            image_tiles = take_strided_stacks_from_single_image(slice,
                                                                   strides=strides,
                                                                   nb_rows=self.params['tile_size'][0],
                                                                   nb_cols=self.params['tile_size'][1],
                                                                   seq_length=nb_slices,
                                                                   recurrent=True,
                                                                   channels = self.params['channels'])
            image_tiles = image_tiles.astype(float)
        else:
            image_tiles = slice.reshape((nb_slices, 1, rows, cols))

        image_tiles = self.normalize(image_tiles)
        if K.image_dim_ordering() == 'th':
            image_tiles = np.pad(image_tiles, ((0, 0), (0, 0), (0, 0), (self.spatial_pad_size, self.spatial_pad_size), (self.spatial_pad_size, self.spatial_pad_size)), 'constant')
        elif K.image_dim_ordering() == 'tf':
            image_tiles = np.pad(image_tiles, ((0, 0), (0, 0), (self.spatial_pad_size, self.spatial_pad_size), (self.spatial_pad_size, self.spatial_pad_size), (0, 0)), 'constant')

        predicted = self.blank_tile_value*np.ones((image_tiles.shape[0], nb_slices, nb_output_features,
                                                   self.unpadded_tile_size[0], self.unpadded_tile_size[1]))

        for tile in range(image_tiles.shape[0]):

            t = image_tiles[[tile, ], :, :, :, :]

            if self.ignore_blank_tiles:
                keep_list, throw_list = self._check_for_blanks(t, threshold=0.8, rate=0.001)
            else:
                keep_list = range(image_tiles.shape[1])

            t = t[:, keep_list, :, :, :]

            p = np.zeros((1, len(keep_list), nb_output_features, self.unpadded_tile_size[0], self.unpadded_tile_size[1]))
            if keep_list:
                for batch, batch_idx in self._batch_generator(t):

                    if self.params['conv_3d']:
                        batch = batch.transpose((0, 2, 3, 1, 4))
                        if batch.shape[3] < self.slice_batch_size:
                            pad = self.slice_batch_size-batch.shape[3]
                            batch = np.pad(batch, ((0,0), (0,0), (0,0), (0, pad), (0,0)), 'constant')
                        else:
                            pad = 0

                    tmp = self.network.predict(batch, verbose=verbose, batch_size=1)

                    if self.params['conv_3d']:
                        if pad > 0:
                            tmp = tmp[:,:,:,:-pad, :]

                        tmp = tmp.transpose((0, 3, 1, 2, 4))

                    if isinstance(tmp, list):
                        tmp = tmp[0]

                    if K.image_dim_ordering() == 'tf':
                        tmp = tmp.transpose((0, 1, 4, 2, 3))

                    self.network.reset_states()

                    tmp = tmp[:, :, :, self.s_pad_l:self.s_pad_u, self.s_pad_l:self.s_pad_u]
                    tmp = tmp[:,self.t_pad_l:self.t_pad_u,[self.output_feature],:,:]
                    p[:, batch_idx[self.t_pad_l:self.t_pad_u], :, :, :] = tmp


                self.network.reset_states()
                for i,j in enumerate(keep_list):
                    predicted[tile, j, :, :, :] = p[:, i, :, :, :]

        predicted = predicted.transpose((0, 3, 4, 1, 2))

        if tile_image:
            reconstructed_slice = reconstruct_feature(tiles=predicted, image_dims=(rows, cols, nb_slices, nb_output_features),
                                                    tile_size=(self.unpadded_tile_size[0], self.unpadded_tile_size[1], nb_slices,
                                                    nb_output_features), stride_length=self.stride_length)
        else:
            reconstructed_slice = predicted

        if self.save_debug_images:
            plt.imsave("debugslice_output.png",255*reconstructed_slice)
            plt.imsave("debugslice_endo.png",255*slice[:,:,0,0])
            plt.imsave("debugslice_tumour.png",255*slice[:,:,0,1])

        if self.is_logit:
            return 255*sigmoid(reconstructed_slice)
        else:
            return 255*reconstructed_slice[:,:,:,0]

    def pad_batches(self, batches, stack, pad_size):

        new_batches = []
        for b in batches:
            new_batch = list(b)
            for p in range(pad_size):
                min_slice = max(new_batch[0]-1,0)
                max_slice = min(new_batch[-1]+1, stack.shape[1]-1)
                new_batch.insert(0, min_slice)
                new_batch.append(max_slice)

            new_batches.append(new_batch)

        return new_batches

    @property
    def architecture_dict(self):
        return {'deep_lstm': DeepConvLSTMModel, 'deep_unet_lstm':DeepUNetLSTMModel}

    def _batch_generator(self, sub_stack):

        if self.slice_batch_size == 'all':
            slice_batch_size = sub_stack.shape[1]
        else:
            slice_batch_size = min(self.slice_batch_size, sub_stack.shape[1])

        slice_batch_splits = range(slice_batch_size, sub_stack.shape[1], slice_batch_size)
        slice_batches = np.split(range(sub_stack.shape[1]), slice_batch_splits)

        slice_batches = self.pad_batches(slice_batches, sub_stack, self.temporal_pad_size)

        for slices in slice_batches:
            yield sub_stack[:, slices, :, :, :], slices


class JobScheduler(object):

    def __init__(self):
        self.jobs = []
        self.current_job = None

    @property
    def nb_jobs(self):
        return len(self.jobs)

    def get_job(self):

        if len(self.jobs) == 0:
            print("Currently no queued jobs.")
            return None

        for j in self.jobs:
            if j.completed:
                self.jobs.remove(j)

        for j in self.jobs:
            if j.started == False and j.completed == False:
                return j

    def add_job(self, job):
        self.jobs.append(job)

    def save_jobs(self, write_location):
        f = open(write_location, "wb")
        dump(self.jobs, f)

    def load_jobs(self, read_location):
        f = open(read_location, "r")
        jobs = load(f)
        self.jobs = jobs


class Job(object):

    def __init__(self, function, args=None, kwargs=None):
        # Takes a function, a List of arguments and a Dict of keyword arguments to be executed at a later date.
        self.args = args
        self.kwargs = kwargs
        self.function = function
        self.started = False
        self.completed = False

    def run(self):
        self.started = True
        try:
            func_val = self.function(*self.args, **self.kwargs)
        except KeyboardInterrupt as e:
            self.started = False
            self.completed = False
            raise e

        self.completed = True
        return func_val


class RollingArray(object):

    def __init__(self, array_size=100):
        self.data = np.zeros(array_size)
        self.size = array_size
        self.cur_loc = 0

    def append(self, value):
        self.data[self.cur_loc] = value
        self.cur_loc += 1
        self.cur_loc %= self.size

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self.data[(self.cur_loc+ii) % self.size] for ii in xrange(*item.indices(len(self)))]

        item_loc = (self.cur_loc + item) % self.size
        return self.data[item_loc]


def _model_dict():
    m_d = dict(unet1=unet1, unet2=unet2, unet3=unet3, unet4=unet4, mini_net=mini_net, mini_net_skeleton=mini_net_skeleton,
               deep_unet_lstm=DeepUNetLSTMModel, mini_lstm=miniConvLSTMModel, mini_skip_lstm=miniSkipConvLSTMModel,
               unet2_skeleton=unet2_skeleton, mini_lstm_skeleton=miniConvLSTMModel_skeleton,
               deep_lstm_skeleton=DeepUNetLSTMModel_skeleton, lstm_skeleton=ConvLSTMModel_skeleton, lstm=ConvLSTMModel,
               full_cnn_convlstm=FullUNetCNNConvLSTM, deep_cnn_convlstm=DeepUNetCNNConvLSTM,
               shallow_cnn_convlstm=ShallowUNetCNNConvLSTM, unet_3d_gen=unet_3d_generator, unet_3d=unet_3d)
    return m_d


def _get_model(architecture_name):
    m_d = _model_dict()
    model = m_d[architecture_name]
    model_params = {}

    if architecture_name in ['deep_unet_lstm', 'mini_skip_lstm', 'mini_lstm', 'mini_lstm_skeleton', 'deep_lstm_skeleton',
                             'lstm', 'lstm_skeleton']:
        model_params['stride_length'] = [128, 128, 5]
        model_params['seq_len'] = 10
        model_params['bidirectional'] = True
    else:
        model_params['channels'] = (2, 1)

    if 'lstm' in architecture_name:
        model_params['recurrent'] = True
    else:
        model_params['recurrent'] = False

    if 'skeleton' in architecture_name:
        model_params['is_logit'] = True
        model_params['output_feature'] = 3

    return model, model_params

def _default_params():

    data_location = '/data/unet/data/'
    training_cases = ["case1/", "case2/", "case3/", "case4/", "case5/", "case6/", "case7/", "testcase1/",
                      "teststack1/", "edgecase3/", "blankcase2/", "blankcase4/", "edgecase2/", "Day13_tile1/",
                      "Day13_tile4/"]

    test_cases = ["Day13_tile2/", "Day13_tile3/", "Day13_tile5/", "24_2A_Day_13_tile1/", "24_2A_Day_13_tile2/",
                  "24_2A_Day_13_tile3/", "24_2A_Day_13_tile4/", "24_2A_Day_13_tile5/", ]

    image_name = 'MultiChan.nii'
    mask_name = 'SkeletonRegress.nii'

    output_folder = '/data/unet/outputs/misc/'

    experiment_name = 'exp_default'

    params = dict(nb_epoch=10, verbose=True, data_location=data_location, training_cases=training_cases,
                  augmentation=False, tile_size=[256, 256], stride_length=[128, 128], image_name=image_name,
                  mask_name=mask_name, architecture='unet2', blank_tiles_rate=0.1, final_activation='linear', optimizer='adam',
                  loss='mean_squared_error', flat_output=False, internal_dropout_rate=0.0, final_dropout_rate=0.0,
                  learning_rate=1e-3, init_style='glorot_uniform', output_folder=output_folder,
                  experiment_name=experiment_name, validation_split=0.1, batch_size=4, initial_weights=None,
                  batch_norm=False, channels=(2,1), data_normalization='scale', leaky_activation=False,
                  smoothing_method=None, smoothing_sigma=0, save_graphs=True, pad_target_features=None,
                  blank_tile_value=0, test_cases=test_cases, plot_logs=False, is_logit=False, output_feature=0,
                  load_weights_by_name=False, reporter_loss=None, image_dim_ordering='th', dataset_size=-1,
                  overfit=False, steps_per_epoch=-1, save_weights_every=1, save_best_only=False, conv_3d=False)

    return params


def _parse_params(params_path):
    if type(params_path) is str:
        if params_path[-5:] == '.yaml':
            with open(params_path) as f:
                params = yaml.load(f)
        elif params_path[-5:] == '.json':
            with open(params_path) as f:
                params = json.load(f)
    else:
        params = params_path

    return params
