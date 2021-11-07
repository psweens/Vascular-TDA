from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import InputSpec
from keras.layers import Recurrent, Convolution2D, MaxPooling2D
from keras.layers.core import Masking

from keras.layers.convolutional import conv_output_length
import numpy as np
from theano import tensor as T


class RecurrentConv2D(Masking):
    '''Abstract base class for recurrent layers.
    Do not use in a model -- it's not a functional layer!
    All recurrent layers (GRU, LSTM, SimpleRNN) also
    follow the specifications of this class and accept
    the keyword arguments listed below.
    # Input shape
        5D tensor with shape `(nb_samples, timesteps, channels,rows,cols)`.
    # Output shape
        - if `return_sequences`: 5D tensor with shape
            `(nb_samples, timesteps, channels,rows,cols)`.
        - else, 2D tensor with shape `(nb_samples, channels,rows,cols)`.
    # Arguments
        weights: list of numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, nb_filter), (nb_filter, nb_filter), (nb_filter,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, rocess the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
            is required when using this layer as the first layer in a model.
        input_shape: input_shape
    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.
        **Note:** for the time being, masking is only supported with Theano.
    # TensorFlow warning
        For the time being, when using the TensorFlow backend,
        the number of timesteps used must be specified in your model.
        Make sure to pass an `input_length` int argument to your
        recurrent layer (if it comes first in your model),
        or to pass a complete `input_shape` argument to the first layer
        in your model otherwise.
    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.
        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                a `batch_input_size=(...)` to the first layer in your model.
                This is the expected shape of your inputs *including the batch
                size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.
        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    '''
    input_ndim = 5

    def __init__(self, weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 nb_row=None, nb_col=None, nb_filter=None,
                 dim_ordering=None,
                 input_dim=None, input_length=None, input_shape=None, **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful

        self.nb_row = nb_row
        self.nb_col = nb_col
        self.nb_filter = nb_filter
        self.dim_ordering = dim_ordering

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(RecurrentConv2D, self).__init__(**kwargs)

    def get_output_mask(self, train=False):
        if self.return_sequences:
            return super(RecurrentConv2D, self).get_output_mask(train)
        else:
            return None

    @property
    def output_shape(self):

        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[2+1]
            cols = input_shape[3+1]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1+1]
            cols = input_shape[2+1]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.return_sequences:
            if self.dim_ordering == 'th':
                return (input_shape[0], input_shape[1],
                        self.nb_filter, rows, cols)
            elif self.dim_ordering == 'tf':
                return (input_shape[0], input_shape[1],
                        rows, cols, self.nb_filter)
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        else:
            if self.dim_ordering == 'th':
                return (input_shape[0], self.nb_filter, rows, cols)
            elif self.dim_ordering == 'tf':
                return (input_shape[0], rows, cols, self.nb_filter)
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def step(self, x, states):
        raise NotImplementedError

    def get_constants(self, X, train=False):
        return None

    def get_initial_states(self, X):
        # (samples, timesteps, row, col, filter)
        initial_state = K.zeros_like(X)
        # (samples,row, col, filter)
        initial_state = K.sum(initial_state, axis=1)
        # initial_state = initial_state[::,]
        initial_state = self.conv_step(initial_state, K.zeros(self.W_shape),
                                       border_mode=self.border_mode)

        initial_states = [initial_state for _ in range(2)]
        return initial_states

    def get_output(self, train=False):

        X = self.get_input(train)
        mask = self.get_input_mask(train)
        constants = self.get_constants(X, train)

        assert K.ndim(X) == 5
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitely the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        last_output, outputs, states = K.rnn(self.step, X,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "return_sequences": self.return_sequences,
                  "go_backwards": self.go_backwards,
                  "stateful": self.stateful}
        if self.stateful:
            config['batch_input_shape'] = self.input_shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length

        base_config = super(RecurrentConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMConv2D(RecurrentConv2D):
    '''
    # Input shape
            5D tensor with shape:
            `(samples,time, channels, rows, cols)` if dim_ordering='th'
            or 5D tensor with shape:
            `(samples,time, rows, cols, channels)` if dim_ordering='tf'.
     # Output shape
        if return_sequences=False
            4D tensor with shape:
            `(samples, nb_filter, o_row, o_col)` if dim_ordering='th'
            or 4D tensor with shape:
            `(samples, o_row, o_col, nb_filter)` if dim_ordering='tf'.
        if return_sequences=True
            5D tensor with shape:
            `(samples, time,nb_filter, o_row, o_col)` if dim_ordering='th'
            or 5D tensor with shape:
            `(samples, time, o_row, o_col, nb_filter)` if dim_ordering='tf'.
        where o_row and o_col depend on the shape of the filter and
        the border_mode
        # Arguments
            nb_filter: Number of convolution filters to use.
            nb_row: Number of rows in the convolution kernel.
            nb_col: Number of columns in the convolution kernel.
            border_mode: 'valid' or 'same'.
            sub_sample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
            dim_ordering: "tf" if the feature are at the last dimension or "th"
            stateful : has not been checked yet.
            init: weight initialization function.
                Can be the name of an existing function (str),
                or a Theano function
                (see: [initializations](../initializations.md)).
            inner_init: initialization function of the inner cells.
            forget_bias_init: initialization function for the bias of the
            forget gate.
                [Jozefowicz et al.]
                (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
                recommend initializing with ones.
            activation: activation function.
                Can be the name of an existing function (str),
                or a Theano function (see: [activations](../activations.md)).
            inner_activation: activation function for the inner cells.
    # References
        - [Convolutional LSTM Network: A Machine Learning Approach for
        Precipitation Nowcasting](http://arxiv.org/pdf/1506.04214v1.pdf)
        The current implementation does not include the feedback loop on the
        cells output
    '''
    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', dim_ordering="tf",
                 border_mode="valid", sub_sample=(1, 1),
                 W_regularizer=None, U_regularizer=None, input_shape=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):

        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.border_mode = border_mode
        self.subsample = sub_sample

        assert dim_ordering in {'tf', "th"}, 'dim_ordering must be in {tf,"th}'
        self.dim_ordering = dim_ordering

        kwargs["nb_filter"] = nb_filter
        kwargs["nb_row"] = nb_row
        kwargs["nb_col"] = nb_col
        kwargs["dim_ordering"] = dim_ordering

        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        super(LSTMConv2D, self).__init__(**kwargs)

    def build(self, in_shape):
        input_shape = in_shape
        # input_shape = self.input_shape
        # self.input = K.placeholder(input_shape)

        if self.dim_ordering == 'th':
            # stack_size = self.input_shape[1+1]
            stack_size = in_shape[1+1]
            self.W_shape = (self.nb_filter, stack_size,
                            self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            # stack_size = self.input_shape[3+1]
            stack_size = in_shape[3+1]
            self.W_shape = (self.nb_row, self.nb_col,
                            stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.dim_ordering == 'th':
            self.W_shape1 = (self.nb_filter, self.nb_filter,
                             self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            self.W_shape1 = (self.nb_row, self.nb_col,
                             self.nb_filter, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (nb_filter)
            self.states = [None, None, None, None]

        self.W_i = self.init(self.W_shape)
        self.U_i = self.inner_init(self.W_shape1)
        self.b_i = K.zeros((self.nb_filter,))

        self.W_f = self.init(self.W_shape)
        self.U_f = self.inner_init(self.W_shape1)
        self.b_f = self.forget_bias_init((self.nb_filter,))

        self.W_c = self.init(self.W_shape)
        self.U_c = self.inner_init(self.W_shape1)
        self.b_c = K.zeros((self.nb_filter))

        self.W_o = self.init(self.W_shape)
        self.U_o = self.inner_init(self.W_shape1)
        self.b_o = K.zeros((self.nb_filter,))

        def append_regulariser(input_regulariser, param, regularizers_list):
            regulariser = regularizers.get(input_regulariser)
            if regulariser:
                regulariser.set_param(param)
                regularizers_list.append(regulariser)

        self.regularizers = []
        for W in [self.W_i, self.W_f, self.W_i, self.W_o]:
            append_regulariser(self.W_regularizer, W, self.regularizers)
        for U in [self.U_i, self.U_f, self.U_i, self.U_o]:
            append_regulariser(self.U_regularizer, U, self.regularizers)
        for b in [self.b_i, self.b_f, self.b_i, self.b_o]:
            append_regulariser(self.b_regularizer, b, self.regularizers)

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')

        if self.return_sequences:
            out_row, out_col, out_filter = self.output_shape[2:]
        else:
            out_row, out_col, out_filter = self.output_shape[1:]

        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0],
                                  out_row, out_col, out_filter)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0],
                                  out_row, out_col, out_filter)))
        else:
            self.states = [K.zeros((input_shape[0],
                                    out_row, out_col, out_filter)),
                           K.zeros((input_shape[0],
                                    out_row, out_col, out_filter))]

    def conv_step(self, x, W, b=None, border_mode="valid"):

        conv_out = K.conv2d(x, W, strides=self.subsample,
                            border_mode=border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=(self.input_shape[0],
                                         self.input_shape[2],
                                         self.input_shape[3],
                                         self.input_shape[4]),
                            filter_shape=self.W_shape)
        if b:
            if self.dim_ordering == 'th':
                conv_out = conv_out + K.reshape(b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                conv_out = conv_out + K.reshape(b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        return conv_out

    def conv_step_hidden(self, x, W, border_mode="valid"):
        # This new function was defined because the
        # image shape must be hardcoded

        if self.return_sequences:
            out_row, out_col, out_filter = self.output_shape[2:]
        else:
            out_row, out_col, out_filter = self.output_shape[1:]

        conv_out = K.conv2d(x, W, strides=(1, 1),
                            border_mode=border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=(self.input_shape[0],
                                         out_row, out_col,
                                         out_filter),
                            filter_shape=self.W_shape1)

        return conv_out

    def step(self, x, states):
        assert len(states) == 4
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_W = states[2]
        B_U = states[3]

        x_i = self.conv_step(x * B_W[0], self.W_i, self.b_i,
                             border_mode=self.border_mode)
        x_f = self.conv_step(x * B_W[1], self.W_f, self.b_f,
                             border_mode=self.border_mode)
        x_c = self.conv_step(x * B_W[2], self.W_c, self.b_c,
                             border_mode=self.border_mode)
        x_o = self.conv_step(x * B_W[3], self.W_o, self.b_o,
                             border_mode=self.border_mode)

        # U : from nb_filter to nb_filter
        # Same because must be stable in the ouptut space
        h_i = self.conv_step_hidden(h_tm1, self.U_i * B_U[0],
                                    border_mode="same")
        h_f = self.conv_step_hidden(h_tm1, self.U_f * B_U[1],
                                    border_mode="same")
        h_c = self.conv_step_hidden(h_tm1, self.U_c * B_U[2],
                                    border_mode="same")
        h_o = self.conv_step_hidden(h_tm1, self.U_o * B_U[3],
                                    border_mode="same")

        i = self.inner_activation(x_i + h_i)
        f = self.inner_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.inner_activation(x_o + h_o)
        h = o * self.activation(c)

        return h, [h, c]

    def get_constants(self, X, train=False):
        retain_p_W = 1. - self.dropout_W
        retain_p_U = 1. - self.dropout_U
        if train and (self.dropout_W > 0 or self.dropout_U > 0):
            nb_samples = K.shape(X)[0]
            if K._BACKEND == 'tensorflow':
                if not self.input_shape[0]:
                    raise Exception('For RNN dropout in tensorflow, ' +
                                    'a complete input_shape must be ' +
                                    'provided (including batch size).')
                nb_samples = self.input_shape[0]
            B_W = [K.random_binomial((nb_samples, self.input_dim),
                                     p=retain_p_W) for _ in range(4)]
            B_U = [K.random_binomial((nb_samples, self.output_dim),
                                     p=retain_p_U) for _ in range(4)]
        else:
            B_W = np.ones(4, dtype=K.floatx()) * retain_p_W
            B_U = np.ones(4, dtype=K.floatx()) * retain_p_U
        return [B_W, B_U]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'border_mode': self.border_mode,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(LSTMConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvRNN(Recurrent):
    """RNN with all connections being convolutions:
    H_t = activation(conv(H_tm1, W_hh) + conv(X_t, W_ih) + b)
    with H_t and X_t being images and W being filters.

    We use Keras' RNN API, thus input and outputs should be 3-way tensors.
    Assuming that your input video have frames of size
    [nb_channels, nb_rows, nb_cols], the input of this layer should be reshaped
    to [batch_size, time_length, nb_channels*nb_rows*nb_cols]. Thus, you have to
    pass the original images shape to the ConvRNN layer.

    Parameters:
    -----------
    filter_dim: list [nb_filters, nb_row, nb_col] convolutional filter
        dimensions
    reshape_dim: list [nb_channels, nb_row, nb_col] original dimensions of a
        frame.
    batch_size: int, batch_size is useful for TensorFlow backend.
    time_length: int, optional for Theano, mandatory for TensorFlow
    subsample: (int, int), just keras.layers.Convolutional2D.subsample

    """
    def __init__(self, filter_dim, reshape_dim,
                 batch_size=None, subsample=(1, 1),
                 init='glorot_uniform', inner_init='glorot_uniform',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, **kwargs):
        self.batch_size = batch_size
        self.border_mode = 'same'
        self.filter_dim = filter_dim
        self.reshape_dim = reshape_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.initial_weights = weights

        self.subsample = tuple(subsample)
        self.output_dim = (filter_dim[0], reshape_dim[1]//self.subsample[0],
                           reshape_dim[2]//self.subsample[1])

        super(ConvRNN, self).__init__(**kwargs)

    def _get_batch_size(self, X):
        if K._BACKEND == 'theano':
            batch_size = X.shape[0]
        else:
            batch_size = self.batch_size
        return batch_size

    def build(self, input_shapes):
        if K._BACKEND == 'theano':
            batch_size = None
        else:
            batch_size = None  # self.batch_size
        input_dim = self.input_shape
        bm = self.border_mode
        reshape_dim = self.reshape_dim
        hidden_dim = self.output_dim

        nb_filter, nb_rows, nb_cols = self.filter_dim
        self.input = K.placeholder(shape=(batch_size, input_dim[1], input_dim[2]))

        # self.b_h = K.zeros((nb_filter,))
        self.conv_h = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)
        self.conv_x = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)

        # hidden to hidden connections
        self.conv_h.build()
        # input to hidden connections
        self.conv_x.build()

        self.max_pool = MaxPooling2D(pool_size=self.subsample, input_shape=hidden_dim)
        self.max_pool.build()

        self.trainable_weights = self.conv_h.trainable_weights + self.conv_x.trainable_weights

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_initial_states(self, X):
        batch_size = self._get_batch_size(X)
        hidden_dim = np.prod(self.output_dim)
        if K._BACKEND == 'theano':
            h = T.zeros((batch_size, hidden_dim))
        else:
            h = K.zeros((batch_size, hidden_dim))
        return [h, ]

    def step(self, x, states):
        batch_size = self._get_batch_size(x)
        input_shape = (batch_size, ) + self.reshape_dim
        hidden_dim = (batch_size, ) + self.output_dim
        nb_filter, nb_rows, nb_cols = self.output_dim
        h_tm1 = K.reshape(states[0], hidden_dim)

        x_t = K.reshape(x, input_shape)
        Wx_t = self.conv_x(x_t, train=True)
        h_t = self.activation(Wx_t + self.conv_h(h_tm1, train=True))
        h_t = K.batch_flatten(h_t)
        return h_t, [h_t, ]

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1], np.prod(self.output_dim))
        else:
            return (input_shape[0], np.prod(self.output_dim))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "filter_dim": self.filter_dim,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "return_sequences": self.return_sequences,
                  "reshape_dim": self.reshape_dim,
                  "go_backwards": self.go_backwards}
        base_config = super(ConvRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvGRU(ConvRNN):
    """ConvGRU is similar to ConvRNN, but with a GRU like state transition
    For documentation and details check seya.layers.conv_rnn.ConvRNN and
    keras.layers.recurrent.GRU

    """
    def __init__(self, filter_dim, reshape_dim, batch_size=None,
                 subsample=(1, 1),
                 init='glorot_uniform', inner_init='glorot_uniform',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, **kwargs):
        super(ConvGRU, self).__init__(
            filter_dim=filter_dim, reshape_dim=reshape_dim,
            batch_size=batch_size, subsample=subsample,
            init=init, inner_init=inner_init, activation=activation,
            inner_activation=inner_activation, weights=weights, **kwargs)

    def build(self, input_shapes):
        batch_size = None
        input_dim = input_shapes
        bm = self.border_mode
        reshape_dim = tuple(self.reshape_dim)
        hidden_dim = tuple(self.output_dim)

        self.input_spec = [InputSpec(shape=input_shapes)]
        self.input_dim = input_shapes[2]

        nb_filter, nb_rows, nb_cols = self.filter_dim
        #self.input = K.placeholder(shape=(batch_size, input_dim[1], input_dim[2]))

        self.b_h = K.zeros((nb_filter,))
        self.b_r = K.zeros((nb_filter,))
        self.b_z = K.zeros((nb_filter,))

        self.conv_h = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)
        self.conv_z = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)
        self.conv_r = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=hidden_dim)

        self.conv_x_h = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)
        self.conv_x_z = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)
        self.conv_x_r = Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=bm, input_shape=reshape_dim)

        # hidden to hidden connections
        self.conv_h.build((nb_filter, hidden_dim[0], nb_rows, nb_cols))
        self.conv_z.build((nb_filter, hidden_dim[0], nb_rows, nb_cols))
        self.conv_r.build((nb_filter, hidden_dim[0], nb_rows, nb_cols))
        # input to hidden connections
        self.conv_x_h.build((nb_filter, reshape_dim[0], nb_rows, nb_cols))
        self.conv_x_z.build((nb_filter, reshape_dim[0], nb_rows, nb_cols))
        self.conv_x_r.build((nb_filter, reshape_dim[0], nb_rows, nb_cols))

        self.max_pool = MaxPooling2D(pool_size=self.subsample, input_shape=reshape_dim)

        self.trainable_weights = self.conv_h.trainable_weights + self.conv_z.trainable_weights + self.conv_r.trainable_weights + \
            self.conv_x_h.trainable_weights + self.conv_x_z.trainable_weights + self.conv_x_r.trainable_weights

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        batch_size = self._get_batch_size(x)
        input_shape = (batch_size, ) + tuple(self.reshape_dim)
        hidden_dim = (batch_size, ) + tuple(self.output_dim)
        nb_filter, nb_rows, nb_cols = self.output_dim
        h_tm1 = K.reshape(states[0], hidden_dim)

        x_t = K.reshape(x, input_shape)
        xz_t = self.conv_x_z(x_t)
        xr_t = self.conv_x_r(x_t)
        xh_t = self.conv_x_h(x_t)

        # xz_t = apply_layer(self.max_pool, xz_t)
        # xr_t = apply_layer(self.max_pool, xr_t)
        # xh_t = apply_layer(self.max_pool, xh_t)

        #xz_t = self.max_pool(xz_t)
        #xr_t = self.max_pool(xr_t)
        #xh_t = self.max_pool(xh_t)

        z = self.inner_activation(xz_t + self.conv_z(h_tm1))
        r = self.inner_activation(xr_t + self.conv_r(h_tm1))
        hh_t = self.activation(xh_t + self.conv_h(r * h_tm1))
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t = K.batch_flatten(h_t)
        return h_t, [h_t, ]

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], np.prod(self.output_dim))
        else:
            return (input_shape[0], np.prod(self.output_dim))

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1], np.prod(self.output_dim))
        else:
            return (input_shape[0], np.prod(self.output_dim))

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "filter_dim": self.filter_dim,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "return_sequences": self.return_sequences,
                  "reshape_dim": self.reshape_dim,
                  "go_backwards": self.go_backwards}
        base_config = super(ConvGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))