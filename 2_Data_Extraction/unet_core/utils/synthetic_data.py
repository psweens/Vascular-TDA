import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_dilation
from unet_core.utils.data_utils import im_smooth, add_noise
from skimage.morphology import skeletonize
from skimage.measure import label
import scipy
from skimage.transform import AffineTransform, warp
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm, trange

def generate_touching_blobs(x, y):
    img = np.zeros((100, 100))

    centre_x_1 = np.random.randint(25) + 25
    centre_y_1 = np.random.randint(25) + 25

    theta = np.random.rand()*2*np.pi

    centre_x_2 = centre_x_1 + np.cos(theta)*15
    centre_y_2 = centre_y_1 + np.sin(theta)*15

    mask = (x - centre_x_1) ** 2 + (y - centre_y_1) ** 2 < 100
    img[mask] = 1
    mask = (x - centre_x_2) ** 2 + (y - centre_y_2) ** 2 < 100
    img[mask] = 2

    grad = np.gradient(img)
    grad = np.sqrt(grad[0]**2 + grad[1]**2)
    img[grad>0] = 0

    dist1 = distance_transform_edt(img!=1)
    dist2 = distance_transform_edt(img!=2)
    sigma = 5

    dist = 0.1 + np.exp(-(dist1 + dist2)**2/sigma)

    img[img>0] = 1

    x_t = img
    y_t = distance_transform_edt(x_t)
    y_t /= np.max(y_t)
    return x_t, y_t, dist


def generate_grid(size=100, step=10):

    grid = np.zeros((size, size))
    for i in range(step, size, step):
        grid[i, :] = 1
        grid[:, i] = 1

    return grid


def generate_vessel_dataset(nb_cases):

    x = []
    y = []

    for i in range(nb_cases):
        im, mask = generate_vessels()
        x.append(im.reshape((1,100,100)))
        y.append(mask.reshape((1,100,100)))
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y


def generate_endothelium_dataset(nb_cases, corrupt_mask=False, distance_mask=False, return_skeleton=False, **kwargs):

    x = []
    y = []

    for i in range(nb_cases):
        im, mask = generate_endothelium(corrupt_mask=corrupt_mask, return_skeleton=return_skeleton)

        if distance_mask:
            mask = distance_transform_edt(mask)

        x.append(im.reshape((1,100,100)).astype(float))
        y.append(mask.reshape((1,100,100)).astype(float))

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y

def generate_complex_endo_dataset(nb_cases, corrupt_mask=False, distance_mask=False, return_skeleton=False, **kwargs):

    x = []
    y = []

    for i in range(nb_cases):
        im, mask = generate_complex_endothelium(corrupt_mask=corrupt_mask, return_skeleton=return_skeleton)

        if distance_mask:
            mask = distance_transform_edt(mask)

        x.append(im.reshape((1,100,100)).astype(float))
        y.append(mask.reshape((1,100,100)).astype(float))

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y

def generate_vessels():

    img = np.zeros((100,100))

    nb_starting_points = 10

    v_0 = np.array([0,0], float)
    v_1 = np.array([0,0], float)
    momentum = 0.9

    nb_steps = 500

    for p in range(nb_starting_points):

        x = np.random.randint(90)+10
        y = np.random.randint(90)+10
        img[x,y] = 1
        theta = 2*np.pi*np.random.rand()

        v_0[0] = np.cos(theta)
        v_0[1] = np.sin(theta)

        for t in range(nb_steps):
            theta = 2*np.pi*np.random.rand()

            v_1[0] = np.cos(theta)
            v_1[1] = np.sin(theta)

            v_1[0] = v_1[0]*(1-momentum) + v_0[0]*momentum
            v_1[1] = v_1[1]*(1-momentum) + v_0[1]*momentum

            v_1 /= np.linalg.norm(v_1)

            x += 1.5*v_1[0]
            y += 1.5*v_1[1]

            x_coord = int(round(x))
            y_coord = int(round(y))

            if x_coord < 0 or x_coord >= 100 or y_coord < 0 or y_coord >= 100:
                break

            img[x_coord, y_coord] = 1
            v_0[0], v_0[1] = v_1[0], v_1[1]

    img = binary_dilation(img, structure=np.ones((3,3))).astype(float)

    mask = img[:]

    img = im_smooth(img, 2)**3

    return img, mask


def dilate_skel(args):
    c = args[0]
    thickness = args[1]
    mask = np.zeros(c.shape)
    s1 = sphere_strel(5, pix_dim=[1.,1.,1.])
    dims = c.shape

    for i in range(thickness):
        w = np.where(c)
        min_x = max(min(w[0]) - 5, 0)
        max_x = min(max(w[0]) + 5, dims[0])
        min_y = max(min(w[1]) - 5, 0)
        max_y = min(max(w[1]) + 5, dims[1])
        min_z = max(min(w[2]) - 5, 0)
        max_z = min(max(w[2]) + 5, dims[2])
        mask[min_x:max_x, min_y:max_y, min_z:max_z] = 1
        c = binary_dilation(c, structure=s1, mask=mask)
    return c

def dilate_skel_2d(args):
    c = args[0]
    thickness = args[1]
    mask = np.zeros(c.shape)
    s1 = sphere_strel_2d(3, pix_dim=[1.,1.])
    dims = c.shape

    for i in range(thickness):
        w = np.where(c)
        min_x = max(min(w[0]) - 5, 0)
        max_x = min(max(w[0]) + 5, dims[0])
        min_y = max(min(w[1]) - 5, 0)
        max_y = min(max(w[1]) + 5, dims[1])
        mask[min_x:max_x, min_y:max_y] = 1
        c = binary_dilation(c, structure=s1, mask=mask)
    return c



def generate_endothelium(image_dims=None, corrupt_mask=False, return_skeleton2d=False, make_3d=False, jitter=False,
                         nb_vessels=3, verbose=False, **kwargs):

    jitter_size = 5

    if make_3d:
        if image_dims is None:
            image_dims = [100,100,100]
        pix_dim=[1., 1., 1.]
        radius=10
        s1 = sphere_strel(5, pix_dim=pix_dim)
        s2 = sphere_strel(30, pix_dim=pix_dim)
    else:
        s1 = np.ones((6,6))
        s2 = np.ones((10,10))
        pix_dim = [1.,1.]

    if jitter:
        image_dims[0] += jitter_size*2
        image_dims[1] += jitter_size*2

    if make_3d:
        skel = generate_skeleton_3d(image_dims=image_dims, pix_dim=pix_dim, return_labels=True, verbose=verbose, **kwargs)
        skel, n_comp = scipy.ndimage.measurements.label(skel, structure=np.ones((3, 3, 3)))
    else:
        skel = generate_skeleton(nb_branches=nb_vessels, size=image_dims[0])
        skel, n_comp = scipy.ndimage.measurements.label(skel, structure=np.ones((3, 3)))


    labels = range(1, int(np.max(skel))+1)
    img = np.zeros(skel.shape)
    t = []
    arr = []
    for n in labels:
        t.append(np.random.randint(2, 7))
        arr.append(skel == n)

    dil = []
    for _arg in tqdm(zip(arr, t), desc='Dilation'):
        if make_3d:
            dil.append(dilate_skel(_arg))
        else:
            dil.append(dilate_skel_2d(_arg))

    mask = img[:]
    for d in tqdm(dil):
        d1 = distance_transform_edt((d > 0).astype(float))
        d2 = distance_transform_edt((d == 0).astype(float))
        i = np.exp(-(d1**2/50))*(d>0) + np.exp(-(d2**2/2))*(d==0)

        img[d] = i[d]

    skel = skel > 0

    img = im_smooth(img, sigma=1, smooth_3d=True)
    img /= np.max(img)

    if make_3d:
        subshape = image_dims[:]
        subshape[-1] /= 5
        subshape = [int(x) for x in subshape]

        img2 = np.zeros(subshape)
        mask2 = np.zeros(subshape)
        skel2 = np.zeros(subshape)

        for slice in range(subshape[-1]):
            img2[:,:,slice] = np.mean(img[:, :, slice*5:(slice+1)*5], axis=2)
            mask2[:,:,slice] = np.max(mask[:, :, slice*5:(slice+1)*5], axis=2)
            skel2[:,:,slice] = np.max(skel[:, :, slice*5:(slice+1)*5], axis=2)

        img = img2
        mask = mask2
        skel = skel2

        pix_dim[-1] *= 5

    if corrupt_mask:
        mask = binary_dilation(mask, structure=s2).astype(float)

    # img = add_noise(img, 'pos_gauss', sigma=0.2)
    # img = add_noise(img, 'poisson')
    # img = add_noise(img, 's&p')
    img[img < 0] = 0
    img[img > 1] = 1

    if jitter:
        for i in range(img.shape[2]):
            x = np.random.randint(-5,5)
            y = np.random.randint(-5,5)
            img[:,:,i] = translate_image(img[:,:,i], x, y)
            mask[:,:,i] = translate_image(mask[:,:,i], x, y)
        img = img[jitter_size:-jitter_size, jitter_size:-jitter_size,:]
        mask = mask[jitter_size:-jitter_size, jitter_size:-jitter_size,:]
        skel = skel[jitter_size:-jitter_size, jitter_size:-jitter_size,:]

    if return_skeleton2d:
        skel2d = np.zeros(img.shape)
        for i in range(skel2d.shape[2]):
            skel2d[:,:,i] = skeletonize(mask[:,:,i])
        return img, mask, skel, skel2d

    return img, mask, skel


def translate_image(image, x, y):
    T = np.eye(3)
    T[0,-1] = x
    T[1,-1] = y
    Tr = AffineTransform(T)
    return warp(image=image,inverse_map=Tr.inverse)


def circle_strel(radius, pix_dim=None):
    if pix_dim is None:
        pix_dim = [1.,1.]

    x, y = np.mgrid[0:2*radius+1,0:2*radius+1]
    mask = (pix_dim[0]*(x-radius))**2 + (pix_dim[1]*(y-radius)**2) < radius**2
    return mask


def sphere_strel(radius, pix_dim=None):
    if pix_dim is None:
        pix_dim = [1.,1.,1.]

    x, y, z = np.mgrid[0:2*radius+1,0:2*radius+1,0:2*radius+1]
    mask = (pix_dim[0]*(x-radius))**2 + (pix_dim[1]*(y-radius)**2) + (pix_dim[2]*2*(z-radius)**2) < radius**2
    return mask

def sphere_strel_2d(radius, pix_dim=None):
    if pix_dim is None:
        pix_dim = [1.,1.]

    x, y = np.mgrid[0:2*radius+1,0:2*radius+1]
    mask = (pix_dim[0]*(x-radius))**2 + (pix_dim[1]*(y-radius)**2) < radius**2

    return mask


def generate_complex_endothelium(corrupt_mask=False, return_skeleton=False):

    min_r = 2
    max_r = 7

    skel = generate_skeleton()
    p = np.where(skel == 1)
    perm = np.random.permutation(len(p[1]))

    img = skel[:]
    img = binary_dilation(img, structure=circle_strel(min_r))

    spaces = np.linspace(0, len(p[1]), max_r - min_r).astype(int)
    spaces = spaces[1:-1]

    for i, s in enumerate(np.split(perm, spaces)):
        r = min_r + i
        strel = circle_strel(r)
        p2 = (p[0][s], p[1][s])
        tmp = np.zeros(skel.shape)
        tmp[p2] = 1
        tmp = binary_dilation(tmp, structure=strel)
        img += tmp

    mask = img[:]

    d1 = distance_transform_edt(img)
    d2 = distance_transform_edt(img == 0)

    img = np.exp(-(d1 ** 2 + d2 ** 2) / 5)

    if corrupt_mask:
        mask = binary_dilation(mask, structure=np.ones((10, 10))).astype(float)

    if return_skeleton:
        mask = skeletonize(mask)

    img += (np.random.rand(100 * 100).reshape((100, 100)) * 0.1)
    img[img < 0] = 0

    return img, mask


def generate_skeleton(nb_branches=8, size=100):
    img = np.zeros((size,size))

    nb_starting_points = nb_branches

    v_0 = np.array([0,0], float)
    v_1 = np.array([0,0], float)
    momentum = 0.9

    nb_steps = 500

    for p in range(nb_starting_points):

        x = np.random.randint(int(size*0.9))+int(size*0.1)
        y = np.random.randint(int(size*0.9))+int(size*0.1)
        img[x,y] = 1
        theta = 2*np.pi*np.random.rand()

        v_0[0] = np.cos(theta)
        v_0[1] = np.sin(theta)

        for t in range(nb_steps):
            theta = 2*np.pi*np.random.rand()

            v_1[0] = np.cos(theta)
            v_1[1] = np.sin(theta)

            v_1[0] = v_1[0]*(1-momentum) + v_0[0]*momentum
            v_1[1] = v_1[1]*(1-momentum) + v_0[1]*momentum

            v_1 /= np.linalg.norm(v_1)

            x += 0.5*v_1[0]
            y += 0.5*v_1[1]

            x_coord = int(round(x))
            y_coord = int(round(y))

            if x_coord < 0 or x_coord >= size or y_coord < 0 or y_coord >= size:
                break

            img[x_coord, y_coord] = 1
            v_0[0], v_0[1] = v_1[0], v_1[1]

    return img


def generate_skeleton_3d(image_dims=None, pix_dim=None, nb_vessels=20, return_labels=False, verbose=False):

    from tqdm import tqdm, trange

    if image_dims is None:
        image_dims = (100,100,40)

    if pix_dim is None:
        pix_dim = [1.,1.,5.]

    img = np.zeros(image_dims)

    nb_starting_points = nb_vessels

    v_0 = np.array([0,0,0], float)
    v_1 = np.array([0,0,0], float)
    momentum = 0.9
    z_motion = 0.1

    nb_steps = 500

    dx = int(0.1*image_dims[0])
    dy = int(0.1*image_dims[1])
    dz = int(0.1*image_dims[2])

    label = 1

    for p in trange(nb_starting_points):


        x = np.random.randint(image_dims[0]-dx)+int(dx/2)
        y = np.random.randint(image_dims[1]-dy)+int(dy/2)
        z = np.random.randint(image_dims[2]-dz)+int(dz/2)
        img[x,y,z] = label
        theta = 2*np.pi*np.random.rand()

        v_0[0] = np.cos(theta)
        v_0[1] = np.sin(theta)
        v_0[2] = z_motion*np.random.randn()/pix_dim[2]

        for t in trange(nb_steps):
            theta = 2*np.pi*np.random.rand()

            v_1[0] = np.cos(theta)
            v_1[1] = np.sin(theta)
            v_1[2] = z_motion*np.random.randn()/pix_dim[2]

            v_1[0] = v_1[0]*(1-momentum) + v_0[0]*momentum
            v_1[1] = v_1[1]*(1-momentum) + v_0[1]*momentum
            v_1[2] = v_1[2]*(1-momentum) + v_0[2]*momentum

            v_1 /= np.linalg.norm(v_1)

            x += 0.5*v_1[0]
            y += 0.5*v_1[1]
            z += 0.5*v_1[2]

            x_coord = int(round(x))
            y_coord = int(round(y))
            z_coord = int(round(z))

            if x_coord < 0 or x_coord >= image_dims[0] or y_coord < 0 or y_coord >= image_dims[1] or z_coord < 0 or z_coord >= image_dims[2]:
                break

            img[x_coord, y_coord, z_coord] = label
            v_0[0], v_0[1], v_0[2] = v_1[0], v_1[1], v_1[2]

        label += 1

    return img


def dot_sphere_dataset(nb_cases, image_dim=20):

    x, y = [], []

    for n in range(nb_cases):
        im = np.zeros((image_dim, image_dim, image_dim))
        mask = np.zeros((image_dim, image_dim, image_dim))

        lower = int(0.2*image_dim)
        upper = int(0.8*image_dim)

        seed_x = np.random.randint(lower, upper)
        seed_y = np.random.randint(lower, upper)
        seed_z = np.random.randint(lower, upper)

        im[seed_x, seed_y, seed_z] = 1
        im = binary_dilation(im, structure=np.ones((2,2,2)))
        mask = im_smooth(im.astype(float), sigma=2)
        mask /= np.max(mask)

        x.append(im.astype(float))
        y.append(mask)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y


if __name__ == "__main__":
    from unet_core.io import write_image
    import os
    nb_volumes = 20

    output_folder = '/data/unet/outputs/paper-experiments/test_data/'

    for i in range(10, 10+nb_volumes):
        folder = output_folder + 'synth_test{}/'.format(i)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        img, mask, skel = generate_endothelium(image_dims=[512, 512, 200], make_3d=True, nb_vessels=20, jitter=True)
        write_image(255*img, folder + 'synthetic_endo.nii'.format(i), pixdim=[1., 1., 5.])
        write_image(mask, folder + 'segmentation.nii'.format(i), pixdim=[1., 1., 5.])
        write_image(skel, folder + 'skel_3d.nii'.format(i), pixdim=[1., 1., 5.])