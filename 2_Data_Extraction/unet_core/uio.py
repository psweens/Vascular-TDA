import os
from xml.etree import ElementTree as ETree

import nibabel as nib
import numpy as np
import h5py

try:
    import javabridge
    import bioformats
except ImportError:
    print("Could not import the javabridge/bioformats libraries requried for handing bioformats images. Only Nifti images will be supported")
    pass

class ImageReader(object):

    def __init__(self, image_path, **kwargs):
        self.reader_class = None
        self.assign_reader(image_path)
        self.reader = self.reader_class(image_path, **kwargs)

    def __enter__(self):
        self.reader.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reader.__exit__(exc_type, exc_value, traceback)

    def assign_reader(self, image_path):

        extension = os.path.splitext(image_path)[1]

        if extension in ['.nii', '.gz']:
            self.reader_class = NiftiImageReader
        elif extension in ['.ims', 'tiff', '.tif', '.czi']:
            self.reader_class = BioFormatsImageReader
        elif extension in ['.h5', '.hdf5']:
            self.reader_class = HDF5Reader
        elif extension in ['.npy', '.npz']:
            self.reader_class = NumpyReader
        else:
            raise ValueError('Unsupported image format')

    def get_slice(self, slice):
        return self.reader.get_slice(slice)

    def get_tile(self, loc, size):
        return self.reader.get_tile(loc, size)

    def get_pixdims(self):
        return self.reader.pix_dims

    def get_dims(self):
        return self.reader.image_dims

    @property
    def get_3d_slices(self):
        return self.reader.get_3d_slices

    @property
    def data(self):
        return self.reader.data


class PythonImageReader(object):

    def __init__(self, image_path, get_3d_slices=False, **kwargs):
            self.path=image_path
            self.image_dims = np.zeros(4, dtype='int16')
            self.get_3d_slices = get_3d_slices

    def initialize(self):
        pass

    def get_slice(self, slice):
        pass


class BioFormatsImageReader(PythonImageReader):


    def __init__(self, image_path, image_series=0, keep_vm_open=False, **kwargs):
        """If another BioformatsImageReader class is to be used later on in the program then call with keep_vm_open=True.
        Java-VM can only be opened once per program. """

        PythonImageReader.__init__(self, image_path, **kwargs)
        self.reader = []
        self.metadata = []
        self.image_dims = np.zeros(4, dtype='uint16')
        self.pix_dims = np.zeros(4)
        self.image_series = image_series
        self.keep_vm_open = keep_vm_open

    def __enter__(self):
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        self.reader = bioformats.ImageReader(self.path)
        self.metadatastr = bioformats.get_omexml_metadata(self.path)
        self.metadata= ETree.fromstring(self.metadatastr.encode('utf-8'))
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        bioformats.release_image_reader(self.reader)
        if self.keep_vm_open is not True:
            javabridge.kill_vm()
        print('closed bioformats reader...')

    def initialize(self):
        series_data = self.metadata[self.image_series]

        for child in series_data.iter():
            if child.tag[-6:] == 'Pixels':
                att = child.attrib

        self.image_dims[0] = att["SizeY"]
        self.image_dims[1] = att["SizeX"]
        self.image_dims[2] = att["SizeZ"]
        self.image_dims[3] = att["SizeC"]

        self.pix_dims[0] = att["PhysicalSizeX"]
        self.pix_dims[1] = att["PhysicalSizeY"]

        if self.get_3d_slices:
            self.image_dims[2] -= 2

    def get_slice(self, slice, loc=None, size=None):
        assert slice < self.image_dims[2]

        if loc is not None and size is not None:
            xywh = (loc[1],loc[0],size[1],size[0])
        else:
            xywh = None

        if self.get_3d_slices:
            midslice = self.reader.read(c=None, z=slice+1, series=self.image_series, rescale=False, XYWH=xywh)
            downslice = self.reader.read(c=None, z=slice, series=self.image_series, rescale=False, XYWH=xywh)
            upslice = self.reader.read(c=None, z=slice+2, series=self.image_series, rescale=False, XYWH=xywh)
            self.current_slice = np.concatenate((midslice, upslice, downslice), axis=2)
        else:
            self.current_slice = self.reader.read(c=None, z=slice, series=self.image_series, rescale=False, XYWH=xywh)

        return self.current_slice

    def get_tile(self, loc, size, channels=None):
        loc = np.array(loc).astype(int)
        size = np.array(size).astype(int)
        xywh = (loc[1], loc[0], size[1], size[0])
        nc = int(self.image_dims[3])

        if channels is None:
            channels = range(nc)

        tile = np.zeros((size[0],size[1],self.image_dims[2], len(channels)))

        for i in range(int(self.image_dims[2])):
            slice = self.reader.read(c=None, z=i, series=self.image_series, rescale=False, XYWH=xywh)
            tile[:,:,i,:] = slice

        return tile

    @property
    def data(self):
        return self.get_tile((0,0),(self.image_dims[0], self.image_dims[1]))


class NiftiImageReader(PythonImageReader):

    def __init__(self, image_path, **kwargs):
        PythonImageReader.__init__(self, image_path, **kwargs)
        self.reader = []
        self.metadata = []
        self.image_dims = []
        self.current_slice = []
        self.pix_dims = []

    def __enter__(self):
        self.image = nib.load(self.path)
        self.reader = self.image.get_data()
        self.metadata = self.image.header
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('closed nifti reader...')

    def initialize(self):
        self.image_dims = np.zeros(4)
        dim = self.metadata['dim']
        self.image_dims[0] = dim[1]
        self.image_dims[1] = dim[2]
        self.image_dims[2] = dim[3]
        self.image_dims[3] = dim[4]

        self.pix_dims = np.zeros(3)
        dim = self.metadata['pixdim']
        self.pix_dims[0] = dim[1]
        self.pix_dims[1] = dim[2]
        self.pix_dims[2] = dim[3]

        if self.get_3d_slices:
            self.image_dims[2] -= 2

    def get_slice(self, slice):
        assert slice < self.image_dims[2]

        if self.get_3d_slices:
            midslice = self.reader[:,:,slice+1]
            downslice = self.reader[:,:,slice]
            upslice = self.reader[:,:,slice+2]
            self.current_slice = np.concatenate((midslice, upslice, downslice), axis=2)
        else:
            self.current_slice = self.reader[:,:,slice]

        return self.current_slice

    def get_tile(self, loc, size, channels=None):
        if channels is None:
            return self.reader[loc[0]:(loc[0]+size[0]), loc[1]:(loc[1]+size[1]),:,:]
        else:
            return self.reader[loc[0]:(loc[0]+size[0]), loc[1]:(loc[1]+size[1]),:,channels]

    def get_data(self):
        return self.reader

    @property
    def data(self):
        return self.reader

class HDF5Reader(NiftiImageReader):

    def __init__(self, image_path, pix_dims=None, **kwargs):
        PythonImageReader.__init__(self, image_path, **kwargs)
        self.reader = []
        self.metadata = []
        self.image_dims = []
        self.current_slice = []
        if pix_dims is None:
            pix_dims = np.array([1.0, 1.0, 1.0])
        self.pix_dims = pix_dims

    def __enter__(self):

        with h5py.File(self.path, 'r') as hf:
            self.image = hf['image']
            self.reader = hf['image'][:]

        self.image_dims = np.array(self.reader.shape)
        return self

class NumpyReader(HDF5Reader):

    def __init__(self, image_path, pix_dims=None, **kwargs):
        PythonImageReader.__init__(self, image_path, **kwargs)
        self.reader = []
        self.metadata = []
        self.image_dims = []
        self.current_slice = []
        if pix_dims is None:
            pix_dims = np.array([1.0, 1.0, 1.0])
        self.pix_dims = pix_dims

    def __enter__(self):
        self.image = np.load(self.path).transpose((1,2,0,3))
        self.reader = np.load(self.path).transpose((1,2,0,3))
        self.image_dims = np.array(self.reader.shape)
        return self



class PythonImageWriter():
    def __init__(self, data=None, image_dims=None, write_path=None, affine=None, pixdim=None, dtype='uint8'):
        self.image_dims = image_dims
        self.data = data
        self.write_path = write_path
        self.affine = affine
        self.dtype = dtype
        self.pixdim = pixdim

        if self.affine is None:
            self.affine = np.eye(4)

        if self.data is None and self.image_dims is not None:
            self.data = np.zeros(self.image_dims, dtype=dtype)
        else:
            self.data = self.data.astype(dtype=dtype)

        if self.data is not None and self.image_dims is None:
            self.image_dims = np.array(self.data.shape, dtype='int16')

        if pixdim is None:
            self.pixdim = np.array([1,1,1])
        else:
            self.pixdim = np.array(self.pixdim)

    def set_data(self, data):
        self.data = data

    def set_dims(self, dims):
        self.image_dims = dims

    def set_dtype(self, dtype):
        if dtype in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'float32', 'float64', 'int', 'float']:
            self.dtype = dtype
            self.data = self.data.astype(dtype)
        else:
            raise ValueError('invalid dtype')

    def set_path(self, write_path):
        self.write_path = write_path

    def add_slice(self, slice):
        pass

    def write(self, write_path):
        pass


class NiftiImageWriter(PythonImageWriter):
    def __init__(self, data=None, image_dims=None, write_path=None, affine=None, pixdim=None, dtype='uint8', work_on_disk=False):
        PythonImageWriter.__init__(self, data, image_dims, write_path, affine, pixdim, dtype)
        self.work_on_disk = work_on_disk

        if self.work_on_disk:
            assert write_path is not None, "Cannot work on disk, no write_path specified"
            self.write(write_path=write_path)
            self.data = nib.load(write_path).dataobj

    def write(self, write_path=None):

        if write_path is not None:
            self.write_path = write_path

        assert self.data is not None
        assert self.write_path is not None

        if self.image_dims is None:
            self.image_dims = self.data.shape

        self.data = self.data.astype(self.dtype)

        nifti_obj = nib.Nifti1Image(self.data, self.affine)
        nifti_obj.header['pixdim'][1:4] = self.pixdim

        nifti_obj.to_filename(self.write_path)

    def add_slice(self, slice):
        if self.data is None:
            self.data = slice.astype(self.dtype)
            self.image_dims = np.zeros(4, dtype='int16')
            for i, j in enumerate(slice.shape):
                self.image_dims[i] = j

            if len(slice.shape) == 2:
                self.image_dims = np.array([self.image_dims[0], self.image_dims[1], 1, 1])
            elif len(slice.shape) == 3:
                self.image_dims = np.array([self.image_dims[0], self.image_dims[1], 1, self.image_dims[2]])

            self.data = self.data.reshape(self.image_dims)

        else:
            self.data = np.concatenate((self.data, slice.astype(self.dtype)), axis=2)
            self.image_dims[2] += 1

    def insert_slice(self, slice, index):
        assert 0 <= index < self.image_dims[2]
        self.data[:, :, index] = slice

    def insert_tile(self, tile, loc, mode='replace'):
        assert tile.shape[2] == self.image_dims[2], "invalid tile stack size"
        tile_size = tile.shape[:2]
        if mode == 'replace':
            self.data[loc[0]:loc[0]+tile_size[0],loc[1]:loc[1]+tile_size[1],:] = tile
        if mode == 'add':
            self.data[loc[0]:loc[0]+tile_size[0],loc[1]:loc[1]+tile_size[1],:] += tile

def write_image(image, path, dtype='uint8', pixdim=None):

    if pixdim is None:
        pixdim = [1,1,1]

    NiftiImageWriter(image, pixdim=pixdim, dtype=dtype).write(path)


def load_image(path):
    with ImageReader(path) as reader:
        return reader.reader.data[:]