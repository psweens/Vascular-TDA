import numpy as np
from skimage.filters import gaussian
import itertools
import keras.backend as K


def draw_square(input_im, pos, side_length):
    im = np.zeros(input_im.shape)
    x = round(pos[0])
    y = round(pos[1])
    l = round(side_length / 2)

    im[x - l:x + l, y - l:y + l] = 1

    return im


def get_moving_square_data(seq_length=10, nb_cases=100, return_sequences=False, nb_rows=50, nb_cols=50):
    data = np.zeros((nb_cases, seq_length + 1, nb_rows, nb_cols))
    for case in range(nb_cases):
        v = np.random.randn((2))
        v /= np.linalg.norm(v)
        v *= np.random.rand()
        template = np.zeros((nb_rows, nb_cols))

        init = np.random.randint(25, 26, 2)
        for t in range(seq_length + 1):
            data[case, t, :, :] = draw_square(template, init + t * v, 6)

    if return_sequences:
        X = data[:, :-1, :, :]
        Y = data[:, 1:, :, :]
    else:
        X = data[:, :-1, :, :]
        Y = data[:, -1, :, :]

    X = np.concatenate(
        (np.zeros((1, 3, 1, nb_rows, nb_cols)), X[:, :, np.newaxis, :, :], np.zeros((1, 3, 1, nb_rows, nb_cols))),
        axis=1)
    Y = np.concatenate(
        (np.zeros((1, 3, 1, nb_rows, nb_cols)), Y[:, :, np.newaxis, :, :], np.zeros((1, 3, 1, nb_rows, nb_cols))),
        axis=1)

    return X, Y


def take_stacks_from_images(x, y, nb_cases_per_image, nb_rows, nb_cols, seq_length, sigma=2):
    out_x = []
    out_y = []

    for x_im, y_im in zip(x, y):
        input_X, input_Y, input_Z = x_im.shape[:3]
        for n in range(nb_cases_per_image):
            stack_origin_X = np.random.randint(0, input_X - nb_rows + 1)
            stack_origin_Y = np.random.randint(0, input_Y - nb_cols + 1)
            stack_origin_Z = np.random.randint(0, input_Z - seq_length + 1)

            stack_x = x_im[stack_origin_X:stack_origin_X + nb_rows,
                      stack_origin_Y:stack_origin_Y + nb_cols,
                      stack_origin_Z:stack_origin_Z + seq_length].copy()
            stack_y = y_im[stack_origin_X:stack_origin_X + nb_rows,
                      stack_origin_Y:stack_origin_Y + nb_cols,
                      stack_origin_Z:stack_origin_Z + seq_length].copy()

            if sigma > 0:
                for i in range(seq_length):
                    mj = np.max((np.max(stack_x[:, :, i]), 0.00001))
                    stack_x[:, :, i] = gaussian(stack_x[:, :, i].astype(float) / mj, sigma=sigma) * mj
                    mj = np.max((np.max(stack_y[:, :, i]), 0.00001))
                    stack_y[:, :, i] = gaussian(stack_y[:, :, i].astype(float) / mj, sigma=sigma) * mj

            stack_x = stack_x.transpose((2, 0, 1))
            stack_y = stack_y.transpose((2, 0, 1))
            out_x.append(stack_x.reshape((seq_length, 1, nb_rows, nb_cols)))
            out_y.append(stack_y.reshape((seq_length, 1, nb_rows, nb_cols)))

    out_x = np.stack(out_x, axis=0)
    out_y = np.stack(out_y, axis=0)

    if K.backend() == 'tensorflow':
        out_x = _convert_array(out_x, th_to_tf=True)
        out_y = _convert_array(out_y, th_to_tf=True)

    return out_x, out_y


def take_strided_stacks_from_images(x, y, strides, nb_rows, nb_cols, seq_length, blank_tiles_rate=0.5, recurrent=False):
    out_x = []
    out_y = []

    first = True

    for x_im, y_im in zip(x, y):
        if len(x_im.shape) == 3:
            input_X, input_Y, input_Z = x_im.shape
            nb_features = 1
        elif len(x_im.shape) == 4:
            input_X, input_Y, input_Z, nb_features = x_im.shape[:]
        else:
            raise ValueError('invalid image volume shape, must be 3D/4D')

        x_strides = np.arange(0, input_X - nb_rows + 1, strides[0])
        y_strides = np.arange(0, input_Y - nb_cols + 1, strides[1])
        z_strides = np.arange(0, input_Z - seq_length + 1, strides[2])

        for x1, y1, z1 in itertools.product(x_strides, y_strides, z_strides):
            stack_origin_X = x1
            stack_origin_Y = y1
            stack_origin_Z = z1

            stack_x = x_im[stack_origin_X:stack_origin_X + nb_rows,
                      stack_origin_Y:stack_origin_Y + nb_cols,
                      stack_origin_Z:stack_origin_Z + seq_length].copy()
            stack_y = y_im[stack_origin_X:stack_origin_X + nb_rows,
                      stack_origin_Y:stack_origin_Y + nb_cols,
                      stack_origin_Z:stack_origin_Z + seq_length].copy()

            stack_y = stack_y.reshape(stack_y.shape[:3])

            stack_x = stack_x.reshape(nb_rows, nb_cols, seq_length, nb_features).transpose((2, 3, 0, 1))
            stack_y = stack_y.transpose((2, 0, 1))

            if np.sum(stack_y > 0.1) > 1000 or np.random.rand() < blank_tiles_rate or np.sum(stack_y > 0.9) > 50 or first:
                out_x.append(stack_x.reshape((seq_length, nb_features, nb_rows, nb_cols)))
                out_y.append(stack_y.reshape((seq_length, 1, nb_rows, nb_cols)))
                first = False

    if len(out_x) > 0:
        if recurrent:
            out_x = np.stack(out_x, axis=0)
            out_y = np.stack(out_y, axis=0)
        else:
            out_x = np.concatenate(out_x, axis=0)
            out_y = np.concatenate(out_y, axis=0)

        if K.backend() == 'tensorflow':
            out_x = _convert_array(out_x, th_to_tf=True)
            out_y = _convert_array(out_y, th_to_tf=True)

    return out_x, out_y


def take_strided_stacks_from_single_image(x_im, strides, nb_rows, nb_cols, seq_length, recurrent=False, channels=None):
    out_x = []

    if len(x_im.shape) == 3:
        input_X, input_Y, input_Z = x_im.shape
        nb_features = 1
    elif len(x_im.shape) == 4:
        input_X, input_Y, input_Z, nb_features = x_im.shape[:]
    else:
        raise ValueError('invalid image volume shape, must be 3D/4D')

    if channels is None:
        channels = range(nb_features)

    x_strides = np.arange(0, input_X - nb_rows + 1, strides[0])
    y_strides = np.arange(0, input_Y - nb_cols + 1, strides[1])
    z_strides = np.arange(0, input_Z - seq_length + 1, strides[2])

    for x1, y1, z1 in itertools.product(x_strides, y_strides, z_strides):
        stack_origin_X = x1
        stack_origin_Y = y1
        stack_origin_Z = z1

        stack_x = x_im[stack_origin_X:stack_origin_X + nb_rows,
                  stack_origin_Y:stack_origin_Y + nb_cols,
                  stack_origin_Z:stack_origin_Z + seq_length].copy()

        stack_x = stack_x.reshape(nb_rows, nb_cols, seq_length, nb_features).transpose((2, 3, 0, 1))

        out_x.append(stack_x.reshape((seq_length, nb_features, nb_rows, nb_cols)))

    if recurrent:
        out_x = np.stack(out_x, axis=0)
        out_x = out_x[:, :, channels, :, :]
    else:
        out_x = np.concatenate(out_x, axis=0)
        out_x = out_x[:, channels, :, :]

    if K.backend() == 'tensorflow':
        out_x = _convert_array(out_x, th_to_tf=True)

    return out_x


def _convert_array(array, th_to_tf=True):
    if th_to_tf:
        if array.ndim == 4:
            return array.transpose((0, 2, 3, 1))

        if array.ndim == 5:
            return array.transpose((0, 1, 3, 4, 2))
    else:
        if array.ndim == 4:
            return array.transpose((0, 3, 1, 2))

        if array.ndim == 5:
            return array.transpose((0, 1, 4, 2, 3))


def get_jitter_data(seq_length=10, nb_cases=100, nb_rows=50, nb_cols=50, padding=0):
    X = np.zeros((nb_cases, seq_length, nb_rows, nb_cols))
    Y = np.zeros((nb_cases, seq_length, nb_rows, nb_cols))

    for case in range(nb_cases):
        template = np.zeros((nb_rows, nb_cols))

        init = np.random.randint(20, 30, 2)

        shape = np.random.randint(3)
        if shape == 0:
            radius = 6 * np.ones(seq_length)
        elif shape == 1:
            radius = np.linspace(1, 16, num=seq_length)
        elif shape == 2:
            radius = np.linspace(0, 16, num=seq_length)
            radius = (radius - 8) ** 2
            radius = 2 * np.sqrt(64 - radius) + 1

        for t in range(seq_length):
            jitter = np.random.randn((2))
            jitter /= np.linalg.norm(jitter)
            jitter *= 2
            X[case, t, :, :] = draw_square(template, init + jitter, radius[t])
            Y[case, t, :, :] = draw_square(template, init, radius[t])

    if padding > 0:
        X = np.concatenate((np.zeros((nb_cases, padding, 1, nb_rows, nb_cols)), X[:, :, np.newaxis, :, :],
                            np.zeros((nb_cases, padding, 1, nb_rows, nb_cols))), axis=1)
        Y = np.concatenate((np.zeros((nb_cases, padding, 1, nb_rows, nb_cols)), Y[:, :, np.newaxis, :, :],
                            np.zeros((nb_cases, padding, 1, nb_rows, nb_cols))), axis=1)

    return X, Y


def take_jittered_stack(input_image, seq_length=10, nb_rows=50, nb_cols=50, sigma=2):
    input_X, input_Y, input_Z = input_image.shape[:3]

    stack_origin_X = np.random.randint(0, input_X - nb_rows - 5)
    stack_origin_Y = np.random.randint(0, input_Y - nb_cols - 5)
    stack_origin_Z = np.random.randint(0, input_Z - seq_length - 5)

    stack = input_image[stack_origin_X:stack_origin_X + nb_rows,
            stack_origin_Y:stack_origin_Y + nb_cols,
            stack_origin_Z:stack_origin_Z + seq_length].copy()

    jittered = np.zeros((nb_rows, nb_cols, seq_length))

    for i in range(seq_length):
        jitter = 5 * np.random.randn(2)
        J_x = stack_origin_X + round(jitter[0])
        J_y = stack_origin_Y + round(jitter[1])

        J_x = int(max(0, J_x))
        J_y = int(max(0, J_y))

        J_x = int(min(input_X - nb_rows, J_x))
        J_y = int(min(input_Y - nb_cols, J_y))

        jittered[:, :, i] = input_image[J_x:J_x + nb_rows, J_y:J_y + nb_cols, stack_origin_Z + i].copy()

    if sigma > 0:
        for i in range(seq_length):
            mj = np.max((np.max(jittered[:, :, i]), 0.00001))
            jittered[:, :, i] = gaussian(jittered[:, :, i].astype(float) / mj, sigma=sigma) * mj
            mj = np.max((np.max(stack[:, :, i]), 0.00001))
            stack[:, :, i] = gaussian(stack[:, :, i].astype(float) / mj, sigma=sigma) * mj

    return jittered, stack


def get_jitter_image_data(images, nb_cases_per_image, nb_rows=50, nb_cols=50, seq_length=10):
    X = []
    Y = []

    for image in images:

        for n in range(nb_cases_per_image):
            stack, jittered = take_jittered_stack(input_image=image, seq_length=seq_length, nb_rows=nb_rows,
                                                  nb_cols=nb_cols)
            stack = stack.transpose((2, 0, 1))
            jittered = jittered.transpose((2, 0, 1))
            X.append(stack.reshape((seq_length, 1, nb_rows, nb_cols)))
            Y.append(jittered.reshape((seq_length, 1, nb_rows, nb_cols)))

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)

    return X, Y
