import numpy as np
from itertools import combinations_with_replacement
from scipy import ndimage as ndi
from skimage.filters import frangi as sk_frangi

def absolute_hessian_eigenvalues(nd_array, sigma=1, scale=True):
    """
    Eigenvalues of the hessian matrix calculated from the input array sorted by absolute value.
    :param nd_array: input array from which to calculate hessian eigenvalues.
    :param sigma: gaussian smoothing parameter.
    :param scale: if True hessian values will be scaled according to sigma squared.
    :return: list of eigenvalues [eigenvalue1, eigenvalue2, ...]
    """
    return absolute_eigenvaluesh(compute_hessian_matrix(nd_array, sigma=sigma, scale=scale))


def divide_nonzero(array1, array2):
    """
    Divides two arrays. Returns zero when dividing by zero.
    """
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)


def create_image_like(data, image):
    return image.__class__(data, affine=image.affine, header=image.header)


def absolute_eigenvaluesh(nd_array):
    """
    Computes the eigenvalues sorted by absolute value from the symmetrical matrix.
    :param nd_array: array from which the eigenvalues will be calculated.
    :return: A list with the eigenvalues sorted in absolute ascending order (e.g. [eigenvalue1, eigenvalue2, ...])
    """
    eigenvalues = np.linalg.eigvalsh(nd_array)
    sorted_eigenvalues = sortbyabs(eigenvalues, axis=-1)
    return [np.squeeze(eigenvalue, axis=-1)
            for eigenvalue in np.split(sorted_eigenvalues, sorted_eigenvalues.shape[-1], axis=-1)]


def sortbyabs(a, axis=0):
    """Sort array along a given axis by the absolute value
    modified from: http://stackoverflow.com/a/11253931/4067734
    """
    index = list(np.ix_(*[np.arange(i) for i in a.shape]))
    index[axis] = np.abs(a).argsort(axis)
    return a[index]

def frangi(nd_array, scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=True, compute_3d=True):

    # from https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/_frangi.py#L74
    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)

    if compute_3d:
        filtered_array = np.zeros(sigmas.shape + nd_array.shape)
        if not nd_array.ndim == 3:
            raise(ValueError("Only 3 dimensions is currently supported"))

        if np.any(np.asarray(sigmas) < 0.0):
            raise ValueError("Sigma values less than zero are not valid")


        for i, sigma in enumerate(sigmas):
            eigenvalues = absolute_hessian_eigenvalues(nd_array, sigma=sigma, scale=True)
            filtered_array[i] = compute_vesselness(*eigenvalues, alpha=alpha, beta=beta, c=frangi_c,
                                                   black_white=black_vessels)

        return np.max(filtered_array, axis=0)

    else:
        filtered_array = np.zeros(nd_array.shape)
        for s in range(nd_array.shape[2]):
            filtered_array[:, :, s] = sk_frangi(nd_array[:, :, s],
                                                           scale_range=scale_range,
                                                           scale_step=scale_step,
                                                           beta1=beta,
                                                           beta2=frangi_c,
                                                           black_ridges=False)

        return filtered_array


def sato_filter(nd_array, scale_range=(1, 10), scale_step=2, alpha=1e5, beta=1e5, frangi_c=500, black_vessels=True, compute_3d=True):

    from skimage.filters import sato

    if compute_3d:
        raise NotImplementedError

    else:
        filtered_array = np.zeros(nd_array.shape)
        for s in range(nd_array.shape[2]):
            filtered_array[:, :, s] = sato(nd_array[:, :, s],
                                                           scale_range=scale_range,
                                                           scale_step=scale_step,
                                                           beta1=alpha,
                                                           beta2=beta,
                                                           black_ridges=False)

    return filtered_array


def compute_measures(eigen1, eigen2, eigen3):
    """
    RA - plate-like structures
    RB - blob-like structures
    S - background
    """
    Ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))
    Rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))
    S = np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))
    return Ra, Rb, S


def compute_plate_like_factor(Ra, alpha):
    return 1 - np.exp(np.negative(np.square(Ra)) / (2 * np.square(alpha)))


def compute_blob_like_factor(Rb, beta):
    return np.exp(np.negative(np.square(Rb) / (2 * np.square(beta))))


def compute_background_factor(S, c):
    return 1 - np.exp(np.negative(np.square(S)) / (2 * np.square(c)))


def compute_vesselness(eigen1, eigen2, eigen3, alpha, beta, c, black_white):
    Ra, Rb, S = compute_measures(eigen1, eigen2, eigen3)
    plate = compute_plate_like_factor(Ra, alpha)
    blob = compute_blob_like_factor(Rb, beta)
    background = compute_background_factor(S, c)
    return filter_out_background(plate * blob * background, black_white, eigen2, eigen3)


def filter_out_background(voxel_data, black_white, eigen2, eigen3):
    """
    Set black_white to true if vessels are darker than the background and to false if
    vessels are brighter than the background.
    """
    if black_white:
        voxel_data[eigen2 < 0] = 0
        voxel_data[eigen3 < 0] = 0
    else:
        voxel_data[eigen2 > 0] = 0
        voxel_data[eigen3 > 0] = 0
    voxel_data[np.isnan(voxel_data)] = 0
    return voxel_data

def compute_hessian_matrix(nd_array, sigma=1, scale=True):
    """
    Computes the hessian matrix for an nd_array.
    This can be used to detect vesselness as well as other features.
    In 3D the first derivative will contain three directional gradients at each index:
    [ gx,  gy,  gz ]
    The Hessian matrix at each index will then be equal to the second derivative:
    [ gxx, gxy, gxz]
    [ gyx, gyy, gyz]
    [ gzx, gzy, gzz]
    The Hessian matrix is symmetrical, so gyx == gxy, gzx == gxz, and gyz == gzy.
    :param nd_array: n-dimensional array from which to compute the hessian matrix.
    :param sigma: gaussian smoothing to perform on the array.
    :param scale: if True, the hessian elements will be scaled by sigma squared.
    :return: hessian array of shape (..., ndim, ndim)
    """
    ndim = nd_array.ndim

    # smooth the nd_array
    smoothed = ndi.gaussian_filter(nd_array, sigma=sigma)

    # compute the first order gradients
    gradient_list = np.gradient(smoothed)

    # compute the hessian elements
    hessian_elements = [np.gradient(gradient_list[ax0], axis=ax1)
                        for ax0, ax1 in combinations_with_replacement(range(ndim), 2)]

    if sigma > 0 and scale:
        # scale the elements of the hessian matrix
        hessian_elements = [(sigma ** 2) * element for element in hessian_elements]

    # create hessian matrix from hessian elements
    hessian_full = [[None] * ndim] * ndim

    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(ndim), 2)):
        element = hessian_elements[index]
        hessian_full[ax0][ax1] = element
        if ax0 != ax1:
            hessian_full[ax1][ax0] = element

    hessian_rows = list()
    for row in hessian_full:
        hessian_rows.append(np.stack(row, axis=-1))

    hessian = np.stack(hessian_rows, axis=-2)
    return hessian

