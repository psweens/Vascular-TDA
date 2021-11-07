import keras
import keras.backend as K
import numpy as np
import tensorflow as tf

if K.backend() == 'theano':
    import theano.tensor as T


def reflect_pad(x, width, batch_ndim=1):
    """
    Pad a tensor with a constant value.
    Parameters
    ----------
    x : tensor
    width : int, iterable of int, or iterable of tuple
        Padding width. If an int, pads each axis symmetrically with the same
        amount in the beginning and end. If an iterable of int, defines the
        symmetric padding width separately for each axis. If an iterable of
        tuples of two ints, defines a seperate padding width for each beginning
        and end of each axis.
    batch_ndim : integer
        Dimensions before the value will not be padded.
    """

    # Idea for how to make this happen: Flip the tensor horizontally to grab horizontal values, then vertically to grab vertical values
    # alternatively, just slice correctly
    input_shape = x.shape
    input_ndim = x.ndim

    output_shape = list(input_shape)
    indices = [slice(None) for _ in output_shape]

    if isinstance(width, int):
        widths = [width] * (input_ndim - batch_ndim)
    else:
        widths = width

    for k, w in enumerate(widths):
        try:
            l, r = w
        except TypeError:
            l = r = w
        output_shape[k + batch_ndim] += l + r
        indices[k + batch_ndim] = slice(l, l + input_shape[k + batch_ndim])

    # Create output array
    out = T.zeros(output_shape)

    # Vertical Reflections
    out = T.set_subtensor(out[:, :, :width, width:-width],
                          x[:, :, width:0:-1, :])  # out[:,:,:width,width:-width] = x[:,:,width:0:-1,:]
    out = T.set_subtensor(out[:, :, -width:, width:-width],
                          x[:, :, -2:-(2 + width):-1, :])  # out[:,:,-width:,width:-width] = x[:,:,-2:-(2+width):-1,:]

    # Place X in out
    # out = T.set_subtensor(out[tuple(indices)], x) # or, alternative, out[width:-width,width:-width] = x
    out = T.set_subtensor(out[:, :, width:-width, width:-width], x)  # out[:,:,width:-width,width:-width] = x

    # Horizontal reflections
    out = T.set_subtensor(out[:, :, :, :width],
                          out[:, :, :, (2 * width):width:-1])  # out[:,:,:,:width] = out[:,:,:,(2*width):width:-1]
    out = T.set_subtensor(out[:, :, :, -width:], out[:, :, :, -(width + 2):-(
    2 * width + 2):-1])  # out[:,:,:,-width:] = out[:,:,:,-(width+2):-(2*width+2):-1]

    return out


class ReflectPadding2D(keras.layers.ZeroPadding2D):

    def __init__(self, padding=1, batch_ndim=2):
        super(ReflectPadding2D, self).__init__()
        self.pad_size = padding
        self.batch_ndim=batch_ndim

    def call(self, x, mask=None):
        return reflect_pad(x, self.pad_size, self.batch_ndim)


class KernelFilter(keras.layers.Conv2D):

    def __init__(self, kernel, filters=1, input_dim=1, nb_rows=3, nb_cols=3, **kwargs):
        kernel = kernel.reshape((input_dim, filters, nb_rows, nb_cols))
        kernel = kernel.transpose((2,3,0,1))
        self.neighbour_kernel = kernel
        super(KernelFilter, self).__init__(filters=filters, kernel_size=(nb_rows, nb_rows), **kwargs)

    def build(self, input_shape):
        super(KernelFilter, self).build(input_shape)
        shape = self.get_weights()[0].shape
        self.neighbour_bias = np.zeros(self.filters)
        self.set_weights((self.neighbour_kernel, self.neighbour_bias))
        self.trainable = False
        self.built = True

    def __call__(self, x, mask=None):
        outputs = super(KernelFilter, self).__call__(x)
        return outputs

class VesselnessLayer(keras.layers.Layer):

    def __init__(self, alpha=2, beta=10, gamma=2, **kwargs):

        super(VesselnessLayer, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, x, mask=None):
        gradient_size = 5
        gradient_kernels = self.build_gradient_kernels(gradient_size)

        double_gradient_kernels = self.build_curve_kernels(gradient_kernels)

        gradient_layers = KernelFilter(gradient_kernels, border_mode='valid', input_dim=1, nb_filter=2)(x)
        gradient_layers = ReflectPadding2D(padding=1, batch_ndim=2)(gradient_layers)

        curve_layers = KernelFilter(double_gradient_kernels, border_mode='valid', input_dim=2, nb_filters=4)(gradient_layers)
        curve_layers = ReflectPadding2D(padding=1, batch_ndim=2)(curve_layers)

        derivative_layers = keras.layers.merge([curve_layers, gradient_layers], mode='concat', concat_axis=-1)

        v_layer = keras.layers.Lambda(self.vesselness_lambda, output_shape=self.vesselness_lambda_shape)(derivative_layers)

        return v_layer

    def build_gradient_kernels(self, gradient_size=3):
        HSOBEL_WEIGHTS = np.array([[1., 2., 1.],
                                   [0., 0., 0.],
                                   [-1., -2., -1.]])
        VSOBEL_WEIGHTS = HSOBEL_WEIGHTS.T

        """TRANSPOSED TO CORRECT FOR FILTER FLIPPING"""
        return np.stack((HSOBEL_WEIGHTS.T, VSOBEL_WEIGHTS.T), axis=0).reshape((2,1,3,3))

    def build_curve_kernels(self, gradient_kernels):
        zeros = np.zeros((1, 3, 3))

        g_xx = np.stack((gradient_kernels[0], zeros[:]), axis=0)
        g_xy = np.stack((gradient_kernels[1], zeros[:]), axis=0)
        g_yx = np.stack((zeros[:], gradient_kernels[0]), axis=0)
        g_yy = np.stack((zeros[:], gradient_kernels[1]), axis=0)

        return np.stack((g_xx, g_yy, g_yx, g_xy)).reshape((4,2,3,3))

    def soft_abs(self, E, axis=-1):
        abk = K.abs(E)

        e_max = K.max(abk, axis=axis)
        e_min = K.min(abk, axis=axis)

        s_sign = K.tanh(E)
        max_id = self.soft_max(abk, axis=axis)

        max_sign = K.sum(s_sign * max_id, axis=axis)
        min_sign = K.sum(s_sign * (T.ones(max_id.shape) - max_id), axis=axis)

        e_max = e_max * max_sign
        e_min = e_min * min_sign

        return e_max, e_min

    def smooth_max(self, E, axis=-1):
        exp = K.exp(E)
        return K.sum(E * exp, axis=axis) / K.sum(exp, axis=axis)

    def smooth_min(self, E, axis=-1):
        exp = K.exp(-E)
        return K.sum(E * exp, axis=axis) / K.sum(exp, axis=axis)

    def soft_max(self, E, axis=-1):
        e_x = T.exp(E)
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def vesselness_lambda(self, x):
        output_shape = self.vesselness_lambda_shape(x.shape)

        splits = T.split(x, [2, 2, 2], 3, axis=-1)

        Tr = K.sum(splits[0], axis=-1)
        D = K.prod(splits[0], axis=-1) - K.prod(splits[1], axis=-1)
        G = K.sum(K.pow(splits[2], 2), axis=-1)

        det_val = K.sqrt(K.clip(K.pow(Tr, 2) / 4 - D, K.epsilon(), 1e7))

        E1 = Tr / 2 + det_val
        E2 = Tr / 2 - det_val

        E = T.stack([E1, E2], axis=0)

        E_max, E_min = self.soft_abs(E, axis=0)

        S = K.pow(E_max, 2) + K.pow(E_min, 2)
        R_B = K.abs(E_min) / (K.abs(E_max) + K.epsilon())

        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        V = (T.ones(S.shape) - K.exp(-S / alpha)) * K.exp(-K.pow(R_B / beta, 2)) * K.sigmoid(
            -E_max*100) * K.exp(-G / gamma)


        V = K.clip(V,0,1)
        V = K.reshape(V, output_shape)

        return V

    def vesselness_lambda_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        return tuple(shape)

class SkeletonLayer(keras.layers.Layer):

    def __init__(self, mode='template', **kwargs):
        self.mode = mode
        super(SkeletonLayer, self).__init__(**kwargs)

    def build_neighbour_kernel(self, connectivity=4):
        kernel = np.ones((3, 3))

        if connectivity == 4:
            kernel[1, 1] = 0
            kernel[0, 0] = 0
            kernel[2, 0] = 0
            kernel[0, 2] = 0
            kernel[2, 2] = 0
        else:
            kernel[1, 1] = 0

        return kernel.reshape((1, 1, 3, 3))

    def build_skeleton_kernels(self):
        template1_f = [[1, 1, 1],
                       [0, 0, 0],
                       [0, 0, 0]]

        template1_b = [[0, 0, 0],
                       [0, 1, 0],
                       [1, 1, 1]]

        t1 = np.stack([template1_f, template1_b], axis=0)

        template2_f = [[0, 1, 1],
                       [0, 0, 1],
                       [0, 0, 0]]

        template2_b = [[0, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]]

        t2 = np.stack([template2_f, template2_b], axis=0)

        t3 = self._rot90(t1)
        t4 = self._rot90(t2)
        t5 = self._rot90(t3)
        t6 = self._rot90(t4)
        t7 = self._rot90(t5)
        t8 = self._rot90(t6)

        t = np.stack([t1, t2, t3, t4, t5, t6, t7, t8], axis=0)

        return t

    @staticmethod
    def _rot90(x):
        for i in range(x.shape[0]):
            x[i, :, :] = np.rot90(x[i,:,:])
        return x

    def call(self, x, mask=None):
        return self.__call__(x, mask)

    def __call__(self, x, mask=None):

        def skeleton_lambda(x):
            min = K.min(K.abs(x), axis=-1)
            act = 1-K.sigmoid(10*min)

            return act

        def skeleton_lambda_shape(input_shape):
            shape = list(input_shape)
            shape[-1] = 1
            return tuple(shape)

        if self.mode == 'template':
            neighbour_kernel = self.build_skeleton_kernels()
        elif self.mode == 'neighbour':
            neighbour_kernel = self.build_neighbour_kernel(connectivity=8)
        else:
            raise ValueError("invalid mode: {}. mode=[template, neighbour]".format(self.mode))

        act1 = x
        input1 = act1
        input2 = keras.layers.Lambda(lambda x: 1-x)(act1)

        merge1 = keras.layers.merge([input1, input2], mode='concat', concat_axis=-1)

        neighbours = KernelFilter(neighbour_kernel, border_mode='valid', nb_filters=8, input_dim=2)(merge1)
        neighbours = ReflectPadding2D(padding=1, batch_ndim=2)(neighbours)

        if self.mode == 'template':
            neighbours = keras.layers.Lambda(skeleton_lambda, output_shape=skeleton_lambda_shape)(neighbours)
            neighbours = keras.layers.Reshape(x._keras_shape[1:])(neighbours)

        output = keras.layers.merge([neighbours, x], mode='concat', concat_axis=-1)

        return output

class GumbelSampling(keras.layers.Layer):
    def __init__(self, tau, name=None, **kwargs):
        self.tau = tau
        super(GumbelSampling, self).__init__(**kwargs)

        self.lambda1 = keras.layers.Lambda(lambda x: 1-x, output_shape=lambda x: x)
        self.lambda2 = keras.layers.Lambda(self._make_logits, output_shape=lambda x: x)
        self.lambda3 = keras.layers.Lambda(self._gumbel_sampling, output_shape=self._sampling_lambda_shape)

        if name is not None:
            self.name = name

    def build(self, input_shape):

        self.lambda1.build(input_shape)
        s1 = self.lambda1.get_output_shape_for(input_shape)
        self.lambda2.build(s1)
        s2 = self.lambda2.get_output_shape_for(s1)
        self.lambda3.build(s2)

        self.built = True

    def call(self, x, mask=None):
        return self.__call__(x, mask)

    def __call__(self, x, mask=None):

        inv_x = self.lambda1(x)

        merge1 = keras.layers.merge([x, inv_x], mode='concat', concat_axis=-1)

        logits = self.lambda2(merge1)
        sample = self.lambda3(logits)

        output = keras.layers.merge([sample, logits], mode='concat', concat_axis=-1, name=self.name)

        return output

    def _soft_max(self, E, axis=-1):
        e_x = T.exp(E - K.max(E, axis=axis, keepdims=True))
        return e_x / (e_x.sum(axis=axis, keepdims=True) + K.epsilon())

    def _make_logits(self, x):
        x = K.clip(x, K.epsilon(), 1 - K.epsilon())
        return K.log(x / (1 - x))

    def _gumbel_sampling(self, logits):
        U = K.random_uniform(K.shape(logits), 0, 1)
        y = logits - K.log(-K.log(U + K.epsilon()) + K.epsilon())  # logits + gumbel noise
        y = self._soft_max(y / self.tau)
        return y

    def _sampling_lambda_shape(self, input_shape):
        return input_shape

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = 2

class GumbelSkeletonLayer(SkeletonLayer):
    def __init__(self, tau, name=None, **kwargs):
        self.tau = tau
        super(GumbelSkeletonLayer, self).__init__(**kwargs)

        if self.mode == 'template':
            neighbour_kernel = self.build_skeleton_kernels()
            nb_filters = 8
            input_dim = 2
        elif self.mode == 'neighbour':
            neighbour_kernel = self.build_neighbour_kernel(connectivity=8)
            nb_filters = 1
            input_dim = 1
        else:
            raise ValueError("invalid mode: {}. mode=[template, neighbour]".format(self.mode))

        self.lambda1 = keras.layers.Lambda(lambda x: 1-x, output_shape=lambda x: x)
        self.lambda2 = keras.layers.Lambda(self._make_logits, output_shape=lambda x: x)
        self.lambda3 = keras.layers.Lambda(self._gumbel_sampling, output_shape=self._sampling_lambda_shape)

        self.neighbours1 = KernelFilter(neighbour_kernel, border_mode='same', filters=nb_filters, input_dim=input_dim)
        self.neighbours2 = ReflectPadding2D(padding=1, batch_ndim=2)

        self.lambda4 = keras.layers.Lambda(self._skeleton_lambda, output_shape=self._skeleton_lambda_shape)

        if name is not None:
            self.name = name

    def build(self, input_shape):

        self.lambda1.build(input_shape)
        s1 = self.lambda1.get_output_shape_for(input_shape)
        self.lambda2.build(s1)
        s2 = self.lambda2.get_output_shape_for(s1)
        self.lambda3.build(s2)
        s3 = self.lambda3.get_output_shape_for(s2)
        s3 = list(s3)
        s3[-1] = 2
        s3 = tuple(s3)
        self.neighbours1.build(s3)
        s4 = self.neighbours1.get_output_shape_for(s3)
        self.neighbours2.build(s4)
        s5 = self.neighbours2.get_output_shape_for(s4)
        self.lambda4.build(s5)

        self.built = True

    def call(self, x, mask=None):
        return self.__call__(x, mask)

    def __call__(self, x, mask=None):

        inv_x = self.lambda1(x)

        merge = keras.layers.merge([x, inv_x], mode='concat', concat_axis=-1)

        logits = self.lambda2(merge)
        sample = self.lambda3(logits)

        neighbours = self.neighbours1(sample)
        # neighbours = self.neighbours2(neighbours)

        if self.mode == 'template':
            neighbours = self.lambda4(neighbours)
            # neighbours = keras.layers.Lambda(lambda s: K.reshape(s, x.shape), output_shape=)(neighbours)

        output = keras.layers.merge([neighbours, sample, logits], mode='concat', concat_axis=-1, name=self.name)

        return output

    def _soft_max(self, E, axis=-1):
        e_x = K.exp(E - K.max(E, axis=axis, keepdims=True))
        return e_x / (K.sum(e_x, axis=axis, keepdims=True) + K.epsilon())

    def _make_logits(self, x):
        x = K.clip(x, K.epsilon(), 1-K.epsilon())
        return K.log(x/(1-x))

    def _gumbel_sampling(self, logits):
        U = K.random_uniform(K.shape(logits), 0, 1)
        y = logits -K.log(-K.log(U + K.epsilon()) + K.epsilon())  # logits + gumbel noise
        y = self._soft_max(y / self.tau)
        return y

    def _skeleton_lambda(self, x):
        min = K.min(K.abs(x), axis=-1)
        act = 1 - K.sigmoid(10 * min)
        # shape = self._skeleton_lambda_shape(K.shape(x))
        # act = K.reshape(act, shape)
        act = K.expand_dims(act, -1)
        return act

    def _skeleton_lambda_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        return tuple(shape)

    def _sampling_lambda_shape(self, input_shape):
        return input_shape

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = 5

        return tuple(output_shape)
