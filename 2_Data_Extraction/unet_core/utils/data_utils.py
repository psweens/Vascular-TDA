import nibabel as nib
import numpy as np
import h5py
import scipy
import skimage
from scipy.ndimage import median_filter, distance_transform_edt, maximum_filter
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.morphology import skeletonize, closing

from stack_utils import take_strided_stacks_from_images
# from unet_core.vessel_analysis import VesselTree
#
# def load_images_as_patches(image_folder, image_name, patch_size=32, stride_length=8, debug=False, slices='all', permute=True):
#     image = nib.load(join(image_folder, image_name)).get_data()
#     if slices == 'all':
#         slices = range(image.shape[2])
#
#     if debug is True:
#         image = image[:256, :256, 10:13]
#         slices = range(3)
#     else:
#         image = image[:, :, slices]
#
#     if len(image.shape) < 4:
#         image = image.reshape((image.shape[0],image.shape[1], image.shape[2], 1))
#
#     image_tiles = image4d_to_patches(image, patch_size, stride_length)
#
#     return image_tiles


def threshold_and_process(input_image, threshold=100, median_filter=False, min_object_size=0, do_closing=False, smooth_3d=False, sigma=2):

    seg = input_image

    if median_filter:
        seg = im_smooth(seg, method='median', smooth_3d=smooth_3d, sigma=sigma)

    seg = seg > threshold
    seg = remove_small_objects(seg, min_size=min_object_size, connectivity=3)
    seg = remove_small_holes(seg, min_size=min_object_size, connectivity=3)

    for i in range(input_image.shape[2]):
        seg[:, :, i] = remove_small_holes(seg[:, :, i], min_object_size)

    if do_closing:
        seg = closing(seg, selem=np.ones((2*sigma+1, 2*sigma + 1, 3)))

    seg = seg.astype('uint8')

    return seg


def downsample_volume(input_image, factor):
    if not isinstance(factor,list):
        factor = (factor,factor,1)

    return input_image[::factor[0], ::factor[1], ::factor[2]]


def im_smooth(input_image, sigma=1, smooth_3d=True, method='gaussian'):

    if method not in ['gaussian', 'median']:
        raise ValueError("Invalid smoothing method: {}".format(method))

    dtype = input_image.dtype
    shape = input_image.shape

    input_image = np.squeeze(input_image)

    max_val = np.max(input_image)
    min_val = np.min(input_image)

    if np.abs(max_val - min_val) < 1e-5:
        return input_image

    if method == 'gaussian':
        input_image = input_image.astype(float)
        input_image -= min_val
        input_image /= (max_val-min_val)

    if len(input_image.shape) > 2:

        if smooth_3d:
            if method == 'gaussian':
                input_image = skimage.filters.gaussian(input_image, sigma=sigma)
            if method == 'median':
                input_image = scipy.ndimage.median_filter(input_image, [sigma,sigma,sigma])
        else:
            for i in range(input_image.shape[2]):
                if method == 'gaussian':
                    input_image[:,:,i] = skimage.filters.gaussian(input_image[:,:,i], sigma=sigma)
                if method == 'median':
                    input_image[:,:,i] = scipy.ndimage.median_filter(input_image[:,:,i], [sigma,sigma])

    else:
        if method == 'gaussian':
            input_image = skimage.filters.gaussian(input_image, sigma=sigma)
        if method == 'median':
            input_image = scipy.ndimage.median_filter(input_image, [sigma,sigma])

    if method == 'gaussian':
        input_image *= (max_val-min_val)
        input_image += min_val
        input_image = input_image.astype(dtype)

    input_image = input_image.reshape(shape)

    return input_image


def add_noise(image, noise_typ='gauss', density=0.05, sigma=0.1):

   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   if noise_typ == "pos_gauss":
      row,col,ch= image.shape
      mean = 0
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      gauss[gauss<0] = 0
      gauss *= 2
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = density
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      image[image<0]=0
      image*=255.
      image+=1.
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      noisy-=1.
      noisy/=255.
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy

# def image4d_to_patches(image, patch_size, stride_length):
#     """Expects input image to be of dimensions (nb_rows, nb_cols, nb_slices, nb_channels)"""
#
#     image_slices = np.transpose(image, (2, 3, 0, 1))
#
#     chan_tiles = []
#     for chan in range(image_slices.shape[1]):
#         chan_slices = image_slices[:,[chan],:, :]
#
#         T_slices = tensor.tensor4('T_slices')
#
#         neibs = images2neibs(T_slices, patch_size, stride_length)
#         window_function = theano.function([T_slices], neibs)
#
#         # Function application
#         tiles = window_function(np.array(chan_slices, dtype='float32'))
#         chan_tiles.append(tiles.reshape((tiles.shape[0], 1, patch_size[0], patch_size[1])))
#
#     image_tiles = np.concatenate(chan_tiles, axis=1)
#
#     return image_tiles
#
# def image4d_to_stack_patches(image, patch_size, stride_length):
#
#     image_slices = np.transpose(image, (2, 3, 0, 1))
#
#     chan_tiles = []
#
#     for slice in range(image_slices.shape[0]):
#
#         slices = []
#
#         for chan in range(image_slices.shape[1]):
#             chan_slices = image_slices[[slice], [chan], :, :]
#
#             T_slices = tensor.tensor4('T_slices')
#
#             neibs = images2neibs(T_slices, patch_size, stride_length)
#             window_function = theano.function([T_slices], neibs)
#
#             # Function application
#             tiles = window_function(np.array(chan_slices, dtype='float32'))
#             chan_tiles.append(tiles.reshape((tiles.shape[0], 1, patch_size[0], patch_size[1])))
#
#         image_tiles = np.concatenate(chan_tiles, axis=1)
#
#         slices.append(image_tiles)
#
#     np.stack(slices,axis=0)
#
#     return slices

#
# def load_images(image_folders, image_name, mask_name, patch_size=32, stride_length=5, testing_fration=0.1, debug=False, blank_tiles_rate=1, slices='all', permute=True):
#
#     image = nib.load(join(image_folders, image_name)).get_data()
#     mask = nib.load(join(image_folders, mask_name)).get_data()
#
#     if slices == 'all':
#         slices = range(image.shape[2])
#
#     if debug is True:
#         image = image[:256, :256, 10:13]
#         mask = mask[:256, :256, 10:13]
#         slices = range(3)
#
#     rows, cols = image.shape[0], image.shape[1]
#
#     image_slices = []
#     mask_slices = []
#
#     for i in np.atleast_1d(slices):
#         image_slices.append(np.reshape(image[:, :, i], (1, rows, cols)))
#         mask_slices.append(np.reshape(mask[:, :, i], (1, rows, cols)))
#
#     T_slices = tensor.tensor4('T_slices')
#
#     neibs = images2neibs(T_slices, patch_size, stride_length)
#     window_function = theano.function([T_slices], neibs)
#
#     # Function application
#     image_tiles = window_function(np.array(image_slices))
#     mask_tiles = window_function(np.array(mask_slices))
#
#     num_tiles = image_tiles.shape[0]
#
#     image_tiles = np.reshape(image_tiles, (num_tiles, 1, patch_size[0], patch_size[1]))
#     mask_tiles = np.reshape(mask_tiles, (num_tiles, 1, patch_size[0], patch_size[1]))
#
#     keep_list = []
#
#     for i in range(num_tiles):
#         if np.sum(mask_tiles[i, :, :, :]) > 5 or rand() < blank_tiles_rate:
#             keep_list.append(i)
#
#     image_tiles = image_tiles[keep_list, :, :, :]
#     mask_tiles = mask_tiles[keep_list, :, :, :]
#
#     num_tiles = image_tiles.shape[0]
#
#     num_test = round(num_tiles * testing_fration)
#
#     if permute:
#         perm = permutation(num_tiles)
#
#         img_test_tiles = image_tiles[perm[:num_test], :]
#         img_train_tiles = image_tiles[perm[num_test:], :]
#
#         mask_test_tiles = mask_tiles[perm[:num_test], :]
#         mask_train_tiles = mask_tiles[perm[num_test:], :]
#     else:
#         img_test_tiles = image_tiles[:num_test, :]
#         img_train_tiles = image_tiles[num_test:, :]
#
#         mask_test_tiles = mask_tiles[:num_test, :]
#         mask_train_tiles = mask_tiles[num_test:, :]
#
#     return (img_train_tiles, mask_train_tiles), (img_test_tiles, mask_test_tiles)


def reconstruct_slice(tiles, image_dims, tile_size, stride_length):

    nb_features = tiles.shape[-3]
    image = np.zeros((nb_features,) + image_dims)
    nb_tiles = tiles.shape[0]
    rows, cols = tiles.shape[-2:]

    for i in range(nb_features):
        tiles_chan = tiles[:, i, :, :].reshape((nb_tiles, rows, cols))
        image[i, :, :] = reconstruct_feature(tiles_chan, image_dims, tile_size, stride_length)

    return image

def reconstruct_feature(tiles, image_dims, tile_size, stride_length):
    cumulative_mask = np.zeros(image_dims)
    cumulative_image = np.zeros(image_dims)
    mask = np.ones(tile_size)

    num_strides_x = (image_dims[0] - tile_size[0])/stride_length[0] + 1
    num_strides_y = (image_dims[1] - tile_size[1])/stride_length[1] + 1
    num_strides_x = int(num_strides_x)
    num_strides_y = int(num_strides_y)

    half_tile_x = tile_size[0]//2
    half_tile_y = tile_size[1]//2
    counter = 0

    half_tile_x = int(half_tile_x)
    half_tile_y = int(half_tile_y)

    for i in range(num_strides_x):
        for j in range(num_strides_y):
            x = i*stride_length[0] + half_tile_x
            y = j*stride_length[1] + half_tile_y

            x = int(x)
            y = int(y)

            cumulative_image[x-half_tile_x:x+half_tile_x,y-half_tile_y:y+half_tile_y] += tiles[counter].reshape(tile_size)
            cumulative_mask[x-half_tile_x:x+half_tile_x,y-half_tile_y:y+half_tile_y] += mask
            counter += 1

    cumulative_image /= cumulative_mask

    return cumulative_image

# def make_skeleton(image, compute_3d=False):
#
#     image = np.squeeze(image)
#     v = VesselTree(image, image_dimensions=image.shape, min_branch_length=20)
#
#     if compute_3d:
#         v.analyse_vessels()
#         skeleton_image = v.make_skeleton_image()
#     else:
#         skeleton_image = np.zeros(image.shape)
#         for i in range(image.shape[2]):
#             skeleton_image[:,:,i] = skeletonize((image[:,:,i] > 0).astype(int))
#
#     return skeleton_image


def distance_transform_ves(image, k, do_dist_transform=True, smooth=False, compute_3d=False):

    if len(image.shape) > 3:
        image = image[:,:,:,0]


    dist = image.astype('float32')

    dist = median_filter(dist, size=[11,11,5])

    if do_dist_transform:
        dist_map = dist[:]

        for slice in range(image.shape[2]):
                dist_map[:, :, slice] = distance_transform_edt(dist[:, :, slice])

        if compute_3d is True:
            dist_map += distance_transform_edt(dist, sampling=[0.8,0.8,5])

    else:
            dist_map = image

    if smooth:
        sigma = 5
        mv = np.max(dist_map)

        if mv > 0:
            dist_map = gaussian(dist_map/mv, sigma=[sigma, sigma, sigma * 0.8 / 5])*mv

    if compute_3d:
        max1 = maximum_filter(dist_map, size=(11, 11, 5))
    else:
        max1 = maximum_filter(dist_map, size=(11, 11, 1))
    # max1 = maximum_filter(dist, size=(10, 10, 5))

    if smooth:
            skel = np.exp((dist_map)/(max1 + 0.000001)) - 1
            skel /= (np.exp(1) - 1)
    else:
            skel = np.exp(dist_map / (max1 + 0.000001)) - 1
            skel /= (np.exp(1) - 1)

    skel = skel ** k

    if np.sum(np.isnan(skel)) > 0:
        print('ERROR: nan-values')

    return skel

def get_data_from_params(params, phase='training'):
    x = []
    y = []
    if phase in ['training', 'train']:
        cases = params['training_cases']
    elif phase in ['testing', 'test']:
        cases = params['test_cases']
    else:
        raise ValueError('invalid phase')
    for c in cases:

        data_x = nib.load(params['data_location'] + c + params['image_name']).get_data()
        data_y = nib.load(params['data_location'] + c + params['mask_name']).get_data()

        if len(data_x.shape) == 4:
            data_x = data_x[:, :, :, params['channels']]
        images_x = [data_x.astype(float), ]
        images_y = [data_y.astype(float), ]

        if params['recurrent'] or params['conv_3d']:
            x_im, y_im = take_strided_stacks_from_images(images_x, images_y,
                                                                   strides=params['stride_length'],
                                                                   nb_rows=params['tile_size'][0],
                                                                   nb_cols=params['tile_size'][1],
                                                                   seq_length=params['tile_size'][2],
                                                                   blank_tiles_rate=params['blank_tiles_rate'],
                                                                   recurrent=True)
        else:
            x_im, y_im = take_strided_stacks_from_images(images_x, images_y,
                                                                   strides=params['stride_length'] + [1,],
                                                                   nb_rows=params['tile_size'][0],
                                                                   nb_cols=params['tile_size'][1],
                                                                   seq_length=1,
                                                                   blank_tiles_rate=params['blank_tiles_rate'],
                                                                   recurrent=False)
        if len(x_im) > 0:
            x.append(x_im)
            y.append(y_im)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    return x, y

def _recurrent_to_3d(x):
    if K.backend() == 'tensorflow':
        x = x.transpose((0,2,3,1,4))
    else:
        x = x.transpose((0,2,3,4,1))
    return x
def _3d_to_recurrent(x):

    if K.backend() == 'tensorflow':
        x = x.transpose((0,3,1,2,4))
    else:
        x = x.transpose((0,4,3,2,1))
    return x


"""Auxiliar methods to deal with loading the dataset."""
import os
import random

import numpy as np

""" from keras.preprocessing.image import apply_transform, flip_axis""" 

""" BJS: Apply transform no longer exists, use imageDataGenerator instead, flip_axis also doesn't exist """
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator()

def flip_axis(x,axis):
    x = np.asarray(x).swapaxes(axis,0)
    x = x[::-1,...]
    x = x.swapaxes(0,axis)
    return x

"""        """

"""from keras.preprocessing.image import transform_matrix_offset_center"""
from keras.preprocessing.image import Iterator, load_img, img_to_array

import keras.backend as K


class TwoImageIterator(Iterator):
    """Class to iterate A and B images at the same time."""

    def __init__(self, directory, a_dir_name='A', b_dir_name='B', load_to_memory=False,
                 is_3d=False, is_recurrent=False,
                 is_a_binary=False, is_b_binary=False, is_a_grayscale=False,
                 is_b_grayscale=False, target_size=(256, 256), rotation_range=0.,
                 height_shift_range=0., width_shift_range=0., zoom_range=0.,
                 fill_mode='constant', cval=0., horizontal_flip=False,
                 vertical_flip=False,  dim_ordering='default', N=-1,
                 batch_size=32, shuffle=True, seed=None, nb_targets=1, seq_len=16):
        """
        Iterate through two directories at the same time.

        Files under the directory A and B with the same name will be returned
        at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - load_to_memory: if true, loads the images to memory when creating the
        iterator;
        - is_a_binary: converts A images to binary images. Applies a threshold of 0.5.
        - is_b_binary: converts B images to binary images. Applies a threshold of 0.5.
        - is_a_grayscale: if True, A images will only have one channel.
        - is_b_grayscale: if True, B images will only have one channel.
        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - seed: seed for a random number generator.
        """
        self.directory = directory

        self.a_dir = os.path.join(directory, a_dir_name)
        self.b_dir = os.path.join(directory, b_dir_name)

        a_files = set(x for x in os.listdir(self.a_dir))
        b_files = set(x for x in os.listdir(self.b_dir))
        # Files inside a and b should have the same name. Images without a pair are discarded.
        self.filenames = list(a_files.intersection(b_files))

        self.is_3d = is_3d
        self.is_recurrent = is_recurrent

        self.shuffle = shuffle
        self.nb_targets = nb_targets

        self.seed = seed

        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            if self.shuffle:
                random.shuffle(self.filenames)
            self.filenames = self.filenames[:N]
        self.N = len(self.filenames)
        if self.N == 0:
            raise Exception("""Did not find any pair in the dataset. Please check that """
                            """the names and extensions of the pairs are exactly the same. """
                            """Searched inside folders: {0} and {1}""".format(self.a_dir, self.b_dir))

        self.dim_ordering = dim_ordering

        if self.dim_ordering == 'default':
            self.dim_ordering = K.image_dim_ordering()

        if self.dim_ordering not in ('th', 'default', 'tf'):
            raise Exception('dim_ordering should be one of "th", "tf" or "default". '
                            'Got {0}'.format(self.dim_ordering))

        if is_3d or is_recurrent:
            # if is_recurrent:
            #     self.target_size = (seq_len,) + target_size
            # else:
            #     self.target_size = target_size + (seq_len,)
            self.target_size = (seq_len,) + target_size
        else:
            self.target_size = target_size

        self.is_a_binary = is_a_binary
        self.is_b_binary = is_b_binary
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale

        self.image_shape_a = self._get_image_shape(self.is_a_grayscale)
        self.image_shape_b = self._get_image_shape(True)

        self.data_format = K.image_data_format()

        self.load_to_memory = load_to_memory
        if self.load_to_memory:
            self._load_imgs_to_memory()

        if self.dim_ordering in ('th', 'default'):
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = -1
            self.row_index = 1
            self.col_index = 2

        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]

        super(TwoImageIterator, self).__init__(len(self.filenames), batch_size,
                                               shuffle, seed)


    def _get_image_shape(self, is_grayscale):
        """Auxiliar method to get the image shape given the color mode."""
        if is_grayscale:
            if self.dim_ordering == 'tf':
                return self.target_size + (1,)
            else:
                return (1,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                return self.target_size + (2,)
            else:
                return (2,) + self.target_size

    def _load_imgs_to_memory(self):
        """Load images to memory."""
        if not self.load_to_memory:
            raise Exception('Can not load images to memory. Reason: load_to_memory = False')

        self.a = np.zeros((self.N,) + self.image_shape_a)
        self.b = np.zeros((self.N,) + self.image_shape_b)

        for idx in range(self.N):
            ai, bi = self._load_img_pair(idx, False)
            self.a[idx] = ai
            self.b[idx] = bi

    def _binarize(self, batch):
        """Make input binary images have 0 and 1 values only."""
        bin_batch = batch / 255.
        bin_batch[bin_batch >= 0.5] = 1
        bin_batch[bin_batch < 0.5] = 0
        return bin_batch

    def _normalize_for_tanh(self, batch):
        """Make input image values lie between -1 and 1."""
        tanh_batch = batch - 127.5
        tanh_batch /= 127.5
        return tanh_batch

    def _load_img_pair(self, idx, load_from_memory):
        """Get a pair of images with index idx."""
        if load_from_memory:
            a = self.a[idx]
            b = self.b[idx]
            return a, b

        fname = self.filenames[idx]

        ext = os.path.splitext(fname)[-1]

        if ext in ['.npy', '.npz']:
            a = np.load(os.path.join(self.a_dir, fname))
            b = np.load(os.path.join(self.b_dir, fname))
        if ext in ['.nii']:
            a = nib.load(os.path.join(self.a_dir, fname)).get_data()
            b = nib.load(os.path.join(self.b_dir, fname)).get_data()
        elif ext in ['.h5', '.h5py']:
            with h5py.File(os.path.join(self.a_dir, fname), 'r') as hf:
                a = hf['image'][:]
            with h5py.File(os.path.join(self.b_dir, fname), 'r') as hf:
                b = hf['image'][:]
        elif ext in ['.png', '.jpg']:
            a = load_img(os.path.join(self.a_dir, fname), grayscale=self.is_a_grayscale)
            b = load_img(os.path.join(self.b_dir, fname), grayscale=self.is_b_grayscale)

            a = img_to_array(a, self.data_format)
            b = img_to_array(b, self.data_format)


        # if self.is_recurrent:
        #     a = a.transpose((1,2,0,3))
        #     b = b.transpose((1,2,0,3))

        return a, b

    def _random_transform(self, a, b):
        """
        Random dataset augmentation.

        Adapted from https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        """
        # a and b are single images, so they don't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * a.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * a.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(rotation_matrix, translation_matrix), zoom_matrix)

        h, w = a.shape[img_row_index], a.shape[img_col_index]
        transform_matrix = img_gen.transform_matrix_offset_center(transform_matrix, h, w)
        a = img_gen.apply_transform(a, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        b = img_gen.apply_transform(b, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_col_index)
                b = flip_axis(b, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_row_index)
                b = flip_axis(b, img_row_index)

        return a, b

    def next(self):
        """Get the next pair of the sequence."""
        # Lock the iterator when the index is changed.
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)

        batch_a = np.zeros((current_batch_size,) + self.image_shape_a)
        batch_b = np.zeros((current_batch_size,) + self.image_shape_b)

        for i, j in enumerate(index_array):
            a_img, b_img = self._load_img_pair(j, self.load_to_memory)
            # a_img, b_img = self._random_transform(a_img, b_img)

            batch_a[i] = a_img
            batch_b[i] = b_img

        if self.nb_targets > 1:
            batch_b = [batch_b]*self.nb_targets

        if self.is_3d and not self.is_recurrent:
            batch_a = batch_a.transpose((0,2,3,1,4))
            batch_b = batch_b.transpose((0,2,3,1,4))

        return [batch_a, batch_b]


class TwoImageCrossValidateIterator(TwoImageIterator):

    def __init__(self, directory, n_folds, **kwargs):
        super(TwoImageCrossValidateIterator, self).__init__(directory, **kwargs)
        self.n_folds = n_folds
        self.folds = np.random.randint(n_folds, size=len(self.filenames))
        self.filenames_fold = []
        self.directory = self.directory
        self.iterator_kwargs = kwargs

        self.filenames_train = []
        self.filenames_val = []

        self.N_train = None
        self.N_val = None

        for i in range(self.n_folds):
            self.filenames_fold.append([x for (x, y) in zip(self.filenames, self.folds) if y == i])

        self.current_fold = 0

    def set_fold(self, fold):
        assert fold in range(self.n_folds), 'Invalid fold'
        self.current_fold = fold
        self.assign_folds()

    def assign_folds(self):
        self.filenames_train = []
        self.filenames_val = []

        for i in range(self.n_folds):
            if i == self.current_fold:
                self.filenames_val += self.filenames_fold[i]
            else:
                self.filenames_train += self.filenames_fold[i]

        self.N_train = len(self.filenames_train)
        self.N_val = len(self.filenames_val)

    def train_iterator(self):
        iterator = TwoImageIterator(self.directory, **self.iterator_kwargs)
        iterator.filenames = self.filenames_train
        iterator.N = self.N_train

        super(TwoImageIterator, iterator).__init__(len(iterator.filenames), iterator.batch_size, iterator.shuffle, iterator.seed)

        return iterator

    def val_iterator(self):
        iterator = TwoImageIterator(self.directory, **self.iterator_kwargs)
        iterator.filenames = self.filenames_val
        iterator.N = self.N_val

        super(TwoImageIterator, iterator).__init__(len(iterator.filenames), iterator.batch_size, iterator.shuffle, iterator.seed)

        return iterator


