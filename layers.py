
import numpy as np
import theano
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne import layers
from lasagne.layers import MergeLayer, Layer, Gate


class FeedbackGRULayer(MergeLayer):
    r"""
    lasagne.layers.recurrent.GRULayer(incoming, num_units,
    resetgate=lasagne.layers.Gate(W_cell=None),
    updategate=lasagne.layers.Gate(W_cell=None),
    hidden_update=lasagne.layers.Gate(
    W_cell=None, lasagne.nonlinearities.tanh),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    Gated Recurrent Unit (GRU) Layer
    Implements the recurrent step proposed in [1]_, which computes the output
    by
    .. math ::
        r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
        u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
        c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
        h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    resetgate : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    References
    ----------
    .. [1] Cho, Kyunghyun, et al: On the properties of neural
       machine translation: Encoder-decoder approaches.
       arXiv preprint arXiv:1409.1259 (2014).
    .. [2] Chung, Junyoung, et al.: Empirical Evaluation of Gated
       Recurrent Neural Networks on Sequence Modeling.
       arXiv preprint arXiv:1412.3555 (2014).
    .. [3] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    Notes
    -----
    An alternate update for the candidate hidden state is proposed in [2]_:
    .. math::
        c_t &= \sigma_c(x_t W_{ic} + (r_t \odot h_{t - 1})W_{hc} + b_c)\\
    We use the formulation from [1]_ because it allows us to do all matrix
    operations in a single dot product.
    """
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),

                 hid_init=init.Constant(0.),
                 out_init=init.Constant(0.),

                 update_in=lambda prev, new: new,
                 update_out=lambda prev, new:new,
                 decorate_in=lambda cur, prev:cur,
                 in_to_repr=None,
                 hid_to_out=None,

                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 n_steps=None,
                 grad_clipping=0,
                 unroll_scan=False,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(FeedbackGRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.only_return_final = only_return_final

        self.update_in = update_in
        self.update_out = update_out
        self.decorate_in = decorate_in
        self.in_to_repr = in_to_repr
        self.hid_to_out = hid_to_out

        self.n_steps = n_steps

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        repr_shape = in_to_repr.output_shape
        input_shape = repr_shape

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[1:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)
        out_shape = hid_to_out.output_shape[1:]
        self.out_shape = out_shape

        repr_shape = in_to_repr.output_shape[1:]
        self.repr_shape = repr_shape

        self.out_init = self.add_param(
            out_init, (1,) + out_shape,
            name="out_init",
            trainable=False,
            regularizable=False
        )

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0], self.n_steps) + self.out_shape

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        hid_init = None

        W_in_stacked = T.concatenate(
                 [self.W_in_to_resetgate, self.W_in_to_updategate,
                  self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_previous, hid_previous, out_previous, *args):
            input_new = self.update_in(input_previous, out_previous)
            input_n = input_new

            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            input_n = self.decorate_in(input_n, out_previous)
            input_n = layers.get_output(self.in_to_repr, input_n)

            input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})

            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid

            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)

            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update
            out = self.update_out(out_previous, layers.get_output(self.hid_to_out, hid))
            return input_new, hid, out

        sequences = []
        step_fun = step
        num_batch = input.shape[0]
        hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)
        out_init = T.dot(T.ones((num_batch,) + len(self.out_shape) * (1,) ), self.out_init)

        input_init = input
        non_seqs = [W_hid_stacked]
        non_seqs += [W_in_stacked, b_stacked]

        inp, hid, out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            go_backwards=self.backwards,
            outputs_info=[input_init, hid_init, out_init],
            non_sequences=non_seqs,
            truncate_gradient=self.gradient_steps,
            n_steps=self.n_steps,
            strict=False)[0]
        rest = tuple(range(2, inp.ndim))
        out = out.transpose((1, 0) + rest)
        return out

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(FeedbackGRULayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += layers.get_all_params(self.hid_to_out, **tags)
        params += layers.get_all_params(self.in_to_repr, **tags)
        return params


class TensorDenseLayer(Layer):
    """
    used to perform embeddings on arbitray input tensor
    X : ([0], [1], ...,  T)
    W : (T, E) where E is the embedding size and T is last dim input size
    returns tensordot(X, W) + b which is : ([0], [1], ..., E)
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(TensorDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        num_inputs = self.input_shape[-1]
        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0:-1] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        activation = T.tensordot(input, self.W, axes=[(input.ndim - 1,), (0,)])
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)


if __name__ == '__main__':
    from lasagne.layers import *
    num_hid = 20
    num_in = 2
    x = InputLayer((None, num_in))
    inp = x

    def update_in(prev_inp, prev_out):
        return prev_inp

    hid = InputLayer((None, num_hid))
    hid_to_out = DenseLayer(hid, num_in)

    feat = DenseLayer(x, 256)
    in_to_repr = feat

    x = FeedbackGRULayer(
        x,
        num_units=num_hid,
        update_in=update_in,
        in_to_repr=in_to_repr,
        hid_to_out=hid_to_out,
        n_steps=12)

    fn = theano.function([inp.input_var], get_output(x, inp.input_var))
    x_example = np.ones((100, num_in)) * 100
    x_example = x_example.astype(np.float32)
    print(fn(x_example)[:, -1])
