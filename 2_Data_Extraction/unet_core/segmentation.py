import argparse
import os
import re
import sys
import time
import itertools
import json

import nibabel as nib
import numpy as np
from skimage.filters import gaussian

from unet_core.core import KerasModel
from unet_core.core import LSTMModel
from unet_core.core import _parse_params, _default_params
from unet_core.io import ImageReader, NiftiImageWriter
from unet_core.utils.data_utils import im_smooth, threshold_and_process


def segment_image_from_params(input_image, input_params, ignore_blank_tiles=True, output_feature=0, is_logit=False, most_recent=False):
    input_params = _parse_params(input_params)
    params = _default_params()
    params.update(input_params)

    weights, params_file, experiment_name = get_best_weights_from_folder(params['output_folder'] + params['experiment_name'] + "/",
                                                                         most_recent=most_recent)
    params['initial_weights'] = params['output_folder'] + params['experiment_name'] + "/" + weights

    params['tau'] = 0.0001

    model = KerasModel(params, ignore_blank_tiles=ignore_blank_tiles)

    output_image = input_image

    if len(input_image.shape) > 2:
        for i in range(input_image.shape[2]):
            tmp = model.segment_slice(input_image[:, :, i], output_feature=output_feature, is_logit=is_logit)
            output_image[:, :, i] = tmp
    else:
        tmp = model.segment_slice(input_image, output_feature=output_feature, is_logit=is_logit)
        output_image = tmp.reshape(input_image.shape)

    return output_image


def deep_vessel_segmentation(image_location, skeleton_model, seg_model, params_path, output_dir,
                             pix_dims=None, suffix=None, keep_vm_open=False, lstm_model=None, **kwargs):

    if suffix is None:
        suffix = ''

    skeleton_path = output_dir + "skelImage" + suffix + ".nii"
    seg_path = output_dir + "segImage" + suffix + ".nii"

    skel_image = segment_image(image_location, params_path=params_path, weights_path=skeleton_model,
                               output_path=skeleton_path, smooth_input=True, write_after_slice=False,
                               dtype='uint16', pix_dims=pix_dims, final_activation='relu',
                               save_debug_image=False, keep_vm_open=True, **kwargs)

    if lstm_model is not None:
        tile_size = [256,256]
        stride_length = [256,256]

        lstm_path = output_dir + "lstmImage" + suffix + ".nii"
        skel_image = lstm_refinement(skel_image, lstm_model, output_path=lstm_path,
                                     pix_dims=[0.8,0.8,5], smooth=True, patch_size=tile_size,
                                     stride_length=stride_length, **kwargs)

    seg_image = segment_image(image_location, params_path=params_path, weights_path=seg_model,
                              output_path=seg_path, smooth_input=True, write_after_slice=False,
                              dtype='uint16', pix_dims=pix_dims, final_activation='sigmoid',
                              save_debug_image=False, keep_vm_open=keep_vm_open, **kwargs)

    maxy = np.max(seg_image)
    skel_image = skel_image.astype(float)/maxy
    skel_image *= seg_image.astype(skel_image.dtype)
    writer = NiftiImageWriter(data=skel_image, dtype='uint8', pixdim=pix_dims)
    writer.write(output_dir + "combiImage" + suffix + ".nii")

    return skel_image


def lstm_refinement(image_path, weights_path, pix_dims=None, output_path=None, smooth=False, patch_size=None, stride_length=None, architecture='deep_lstm', **kwargs):

    if isinstance(image_path, str):
        input_image = nib.load(image_path).get_data()
    else:
        input_image = image_path

    if patch_size is None:
        patch_size = [512, 512]

    if stride_length is None:
        stride_length = [512, 512]

    pad_size = 3

    params = {'patch_size': patch_size, 'stride_length': stride_length, 'normalization': 'scale'}
    sequence_padding=5

    model = LSTMModel(params=params, weights=weights_path, spatial_pad_size=pad_size, temporal_pad_size=sequence_padding, architecture=architecture, **kwargs)

    seg_image = model.segment_stack(input_image)

    seg_image[seg_image < 0] = 0
    seg_image[seg_image > 255] = 255

    if smooth:
        seg_image = im_smooth(seg_image, smooth_3d=True)

    if output_path is not None:
        if pix_dims is None:
            pix_dims = [1, 1, 1]

        writer = NiftiImageWriter(seg_image, pixdim=pix_dims)
        writer.write(output_path)

    return seg_image


def segment_image(image_path, weights_path, params_path, output_path=None, dtype='uint8',
                  pix_dims=None, smooth_input=False, sigma=3, write_after_slice=False, final_activation=None, chan_order=(2,1),
                  save_debug_image=False, keep_vm_open=False, debug=False, ignore_blank_tiles=True, aux_params=None,
                  min_object_size=0, threshold=0, post_process=False, render_path=None, show_render=False, model=None, **kwargs):

    core_params = _parse_params(params_path)
    input_params = dict(final_activation=final_activation, initial_weights=weights_path,
                        tile_size=[512,512], stride_length=[512,512], channels=chan_order)

    core_params.update(input_params)
    if aux_params is not None:
        core_params.update(aux_params)

    if output_path is not None and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with ImageReader(image_path, image_series=1, keep_vm_open=keep_vm_open) as reader:

        start_time = time.time()

        if pix_dims is None:
            pix_dims = np.zeros(3)
            pd = reader.get_pixdims()
            pix_dims[0] = pd[0]
            pix_dims[1] = pd[1]
            pix_dims[2] = 5

        image_dims = reader.get_dims().astype(int)

        num_slices = int(image_dims[2])
        writer = NiftiImageWriter(data=np.zeros(image_dims[:3], dtype=dtype), pixdim=pix_dims, dtype=dtype)
        writer.set_dtype(dtype=dtype)
        slices = range(num_slices)

        if core_params['recurrent'] or core_params['conv_3d']:
            if model is None:
                model = LSTMModel(params=core_params, spatial_pad_size=0,
                              temporal_pad_size=0, ignore_blank_tiles=ignore_blank_tiles, **kwargs)
            seg_image = model.segment_stack(reader.reader.data)
            writer.set_data(seg_image)
        else:
            if model is None:
                model = KerasModel(core_params, ignore_blank_tiles=ignore_blank_tiles, save_debug_images=save_debug_image)

            if model.use_3d:
                keep_chans = []
                for s in range(3):
                    for c in chan_order:
                        keep_chans.append(c + s*(len(chan_order)+1))
                chan_order = keep_chans

            for index in slices:
                print('Slice: {0}/{1}'.format(index+1, num_slices))

                slice = reader.get_slice(index)
                if len(slice.shape) == 2:
                    slice = slice.reshape(slice.shape + (1,))

                slice = slice[:, :, core_params['channels']].astype(float)

                if smooth_input:
                    for i in range(slice.shape[2]):
                        max_val = np.max(slice[:,:,i])
                        slice[:,:,i] = gaussian(slice[:,:,i].astype('float')/float(max_val), sigma=sigma) * max_val

                segmented_slice = model.segment_slice(slice)
                writer.insert_slice(segmented_slice, index)
                if write_after_slice is True:
                    writer.write(output_path)

        if output_path is not None:

            if threshold > 0 or post_process:
                writer.data = 255*threshold_and_process(writer.data, threshold=threshold, median_filter=post_process, min_object_size=100)

            writer.write(output_path)

        if render_path is not None or show_render:
            from mayavi import mlab
            data = mlab.pipeline.scalar_field(writer.data[:,:,::-1])
            data.spacing = pix_dims
            mlab.pipeline.volume(data)
            if render_path is not None:
                mlab.savefig(render_path)

            if show_render:
                mlab.show()

    end_time = time.time()
    time_total = end_time - start_time
    print("Segmentation time: {0}".format(time_total))
    return writer.data


def segment_large_image(image_path, weights_path, params_path, output_path, dtype='uint8',
                  pix_dims=None, smooth_input=False, sigma=3, write_after_slice=False, final_activation=None, chan_order=(2,1),
                  save_debug_image=False, keep_vm_open=False, debug=False, ignore_blank_tiles=True, aux_params=None, min_object_size=0, threshold=0, post_process=False, **kwargs):

    core_params = _parse_params(params_path)
    input_params = dict(final_activation=final_activation, initial_weights=weights_path,
                        tile_size=[512,512], stride_length=[512,512], channels=chan_order)

    core_params.update(input_params)
    if aux_params is not None:
        core_params.update(aux_params)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with ImageReader(image_path, image_series=1, keep_vm_open=keep_vm_open) as reader:

        start_time = time.time()

        if pix_dims is None:
            pix_dims = np.zeros(3)
            pd = reader.get_pixdims()
            pix_dims[0] = pd[0]
            pix_dims[1] = pd[1]
            pix_dims[2] = 5

        image_dims = reader.get_dims().astype(int)

        num_slices = int(image_dims[2])
        writer = NiftiImageWriter(data=np.zeros(image_dims[:3], dtype='uint16'), pixdim=pix_dims)
        writer.set_dtype(dtype='uint16')
        slices = range(num_slices)

        print(image_dims)

        if core_params['recurrent']:
            model = LSTMModel(params=core_params, spatial_pad_size=0,
                              temporal_pad_size=0, **kwargs)
            model.calibrate(image_dims)
            tiles = _split_into_tiles(image_dims, model.tile_size, model.stride_length)
            writer_accum = NiftiImageWriter(data=np.zeros(image_dims[:3], dtype=np.uint8), pixdim=pix_dims, dtype='uint8')

            for i, t in enumerate(tiles):
                print("Tile: {}/{}".format(i, len(tiles)))
                tile = reader.get_tile(t, core_params['tile_size'])
                seg_image = model.segment_stack(tile, verbose=True)
                writer.insert_tile(seg_image.astype('uint16'), t, mode='add')
                # writer_accum.insert_tile(np.ones_like(seg_image, dtype=np.uint8), t, mode='add')

            # writer.data /= writer_accum.data

        else:
            model = KerasModel(core_params, ignore_blank_tiles=ignore_blank_tiles, save_debug_images=save_debug_image)

            if model.use_3d:
                keep_chans = []
                for s in range(3):
                    for c in chan_order:
                        keep_chans.append(c + s*(len(chan_order)+1))
                chan_order = keep_chans

            for index in slices:
                print('Slice: {0}/{1}'.format(index+1, num_slices))

                slice = reader.get_slice(index)
                if len(slice.shape) == 2:
                    slice = slice.reshape(slice.shape + (1,))

                slice = slice[:, :, core_params['channels']].astype(float)

                if smooth_input:
                    for i in range(slice.shape[2]):
                        max_val = np.max(slice[:,:,i])
                        slice[:,:,i] = gaussian(slice[:,:,i].astype('float')/float(max_val), sigma=sigma) * max_val

                segmented_slice = model.segment_slice(slice)
                writer.insert_slice(segmented_slice, index)
                if write_after_slice is True:
                    writer.write(output_path)

        if output_path is not None:

            if threshold > 0 or post_process:
                writer.data = 255*threshold_and_process(writer.data, threshold=threshold, median_filter=post_process, min_object_size=100)

            writer.write(output_path)

    end_time = time.time()
    time_total = end_time - start_time
    print("Segmentation time: {0}".format(time_total))
    return writer.data


def _split_into_tiles(image_dims, tile_size, stride_length):
    assert image_dims[0] % tile_size[0] == 0, "invalid tile size in the x-direction, {} does not divide by {}".format(image_dims[0], tile_size[0])
    assert image_dims[1] % tile_size[1] == 0, "invalid tile size in the y-direction, {} does not divide by {}".format(image_dims[1], tile_size[1])

    x_tiles = range(0, image_dims[0]-tile_size[0]+stride_length[0], stride_length[0])
    y_tiles = range(0, image_dims[1]-tile_size[1]+stride_length[1], stride_length[1])
    tiles = itertools.product(x_tiles, y_tiles)
    return list(tiles)


def get_weights_from_folder(folder, epoch):
    correct_file = None
    files = os.listdir(folder)
    model_name = folder.split('/')[-2]
    for file in files:
        regex = re.match("weights[.]?([\d]*)-([+-]?[\d]*.[\d]*).hdf5", file)
        if regex:
            _loss = float(regex.group(2))
            _epoch = float(regex.group(1))

            if _epoch == epoch:
                correct_file = file

    if correct_file is None:
        raise ValueError("No valid weights files were present")

    if not os.path.exists(folder + 'params.json'):
        raise ValueError("No valid params files was present")
    else:
        params = 'params.json'

    return correct_file, params, model_name


def get_best_weights_from_folder(folder, most_recent=False, metric='val_reporter_loss', return_loss=False):

    if folder[-1] is not "/":
        folder += "/"

    if most_recent:
        best_loss = -np.inf
    else:
        best_loss = np.inf

    best_file = None
    files = os.listdir(folder)
    model_name = folder.split('/')[-2]

    if os.path.exists(folder + 'loss.json'):
        with open(folder + 'loss.json') as f:
            losses = json.load(f)

        if not most_recent:
            epoch_nb = np.argmin(losses[metric])
            best_loss = losses[metric][epoch_nb]
        else:
            epoch_nb = len(losses[metric]) - 1
            best_loss = losses[metric][-1]

        epoch_str = "%02d" % epoch_nb
        for file in files:
            regex = re.match("weights." + epoch_str +"-([+-]?[\d]*.[\d]*).hdf5", file)
            if regex:
                best_file = file

    else:
        for file in files:
            regex = re.match("weights[.]?([\d]*)-([+-]?[\d]*.[\d]*).hdf5", file)
            if regex:
                loss = float(regex.group(2))
                epoch = float(regex.group(1))
                if loss < best_loss and not most_recent:
                    best_file = file
                    best_loss = loss
                elif epoch > best_loss and most_recent:
                    best_file = file
                    best_loss = epoch

    if best_file is None:
        raise ValueError("No valid weights files were present")

    if not os.path.exists(folder + 'params.json'):
        raise ValueError("No valid params files was present")
    else:
        params = 'params.json'

    if return_loss:
        return best_file, params, model_name, best_loss
    else:
        return best_file, params, model_name


if __name__ is "__main__":
    parser = argparse.ArgumentParser(description="Apply segmentation method")

    parser.add_argument('image_path', type=str)
    parser.add_argument('--combi', dest='method', action='store_const',
                        const=deep_vessel_segmentation, default=segment_image)

    parser.add_argument('-skel_model', dest='skel_model', default=None)
    parser.add_argument('-seg_model', dest='seg_model', default=None)
    parser.add_argument('-params_path', dest='params_path', default=None)
    parser.add_argument('-output_path', dest='output_path', default=None)
    parser.add_argument('-chan_order', dest='channels', default=(2,1))
    parser.add_argument('--debug', dest='debug', action='store_const',const= True, default=False)
    parser.add_argument('-pix_dims', dest='pix_dims', default=[0.8,0.8,5])
    args = parser.parse_args(sys.argv[1:])

    if args.output_path is None:
        print("Must specify an output path. Use flag -output_path")
        sys.exit()

    if args.seg_model is not None and args.skel_model is not None and args.method == segment_image:
        print("Specified both a seg model and a skel model. Use --combi flag to perform combination method")
        sys.exit()

    if args.seg_model is not None and args.skel_model is None and args.method == segment_image:
        args.method(args.image_path, params_path=args.params_path, weights=args.seg_model, output_path=args.output_path,
                    smooth_input=True, write_after_slice=False, dtype='uint16', pixdim=args.pix_dims,
                    final_activation='sigmoid', chan_order=args.channels, save_debug_image=True, keep_vm_open=True,
                    debug=args.debug)

    if args.seg_model is None and args.skel_model is not None and args.method == segment_image:
        args.method(args.image_path, params_path=args.params_path, weights=args.skel_model,
                    output_path=args.output_path,
                    smooth_input=True, write_after_slice=False, dtype='uint16', pixdim=args.pix_dims,
                    final_activation='relu',
                    chan_order=args.channels, save_debug_image=True, keep_vm_open=True, debug=args.debug)

    if args.seg_model is not None and args.skel_model is not None and args.method == deep_vessel_segmentation:
        args.method(args.image_path, params_path=args.params_path, skeleton_model=args.skel_model,
                    seg_model=args.seg_model,
                    output_dir=args.output_path, channels=(2, 1), pix_dims=args.pix_dims, suffix=None, debug=False,
                    keep_vm_open=False)

    print('Done')
