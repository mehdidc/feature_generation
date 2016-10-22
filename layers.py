
import numpy as np
import theano
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne import layers
from lasagne.layers import MergeLayer, Layer, Gate
import lasagne

class FeedbackGRULayer(MergeLayer):
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 out_init=init.Constant(0.),
                 update_in=lambda prev, new: new, # updated input based to the GRU based on (previous used input=prev, previous predicted output=new)
                 update_out=lambda prev, new:new, # updated GRU output based on (previous output=prev, predicted output=new)
                 decorate_in=lambda cur, prev:cur, # decorate GRU input passed to in_to_repr based on (current input, previous output)
                 in_to_repr=None, # in to repr network mapping
                 hid_to_out=None, # hid to out network mapping
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 n_steps=None,
                 grad_clipping=0,
                 unroll_scan=False,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):
        """
        story:
        the feed back gru layer has an input, an input to repr mapping,
        a hidden to output mapping, a hidden state and an output.
        each time step has input, representation, hidden state and output.
        at each time step t:

        - we take the previous input I_{t-1}, the previous output O_{t-1}, apply update_in(I_{t-1}, O_{t-1})
        to get the new input I_{t}. 
        - grad clip is applied to I_{t} and updated
        - the new input I_{t} is decorated using I_{t} = decorate_in(I_{t}, O_{t-1}). 
        - I_{t} is sent to in_to_repr(I_{t}) to get a representation R_{t}. 
        - R_{t} is the real input which GRU receives, it updates its hidden state and produces H_{t}.
        - the output O_{t} is computed using the network hid_to_out(H_{t}). 
        - the utput is updated  using O_{t} = update_out(O_{t-1}, O_{t}).
        """
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


class FeedbackGRULayerClean(MergeLayer):
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 out_init=init.Constant(0.),
                 predict_input=lambda xprev, oprev, hprev: xprev , # compute new input based to the GRU based on (previous used input, previous hidden state, previous predicted output)
                 predict_repr=lambda x:x,
                 predict_output=lambda oprev, h: oprev, #compute new output based on (previous output, current hidden state)
                 repr_shape=None,
                 out_shape=None,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 n_steps=None,
                 grad_clipping=0,
                 unroll_scan=False,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):
        """
        story:
        the feed back gru layer has an input, an input to repr mapping,
        a hidden to output mapping, a hidden state and an output.
        each time step has input, representation, hidden state and output.
        at each time step t:

        """
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
        super(FeedbackGRULayerClean, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.only_return_final = only_return_final
    
        self.predict_input = predict_input
        self.predict_repr = predict_repr
        self.predict_output = predict_output
        self.n_steps = n_steps

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        assert repr_shape
        input_shape = repr_shape
        assert out_shape
        out_shape = out_shape[1:]
        self.out_shape = out_shape
        self.repr_shape = repr_shape

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
            hid_input = T.dot(hid_previous, W_hid_stacked)
            if self.grad_clipping:
                input_previous = theano.gradient.grad_clip(
                    input_previous, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            input_n = self.predict_input(input_previous, hid_previous, out_previous)
            input_cur = input_n
            input_n = self.predict_repr(input_n)
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
            out = self.predict_output(out_previous, hid)
            return input_cur, hid, out

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
        rest = tuple(range(2, out.ndim))
        out = out.transpose((1, 0) + rest)
        return out

    def get_params(self, **tags):
        params = super(FeedbackGRULayerClean, self).get_params(**tags)
        return params

class AddParams(layers.Layer):

    def __init__(self, incoming, layers, **kwargs):
        super(AddParams, self).__init__(incoming, **kwargs)
        self.layers = layers

    def get_output_for(self, input, **kwargs):
            return input

    def get_params(self, **tags):
        params = [p for l in self.layers for p in layers.get_all_params(l, **tags)]
        return params

class Deconv2DLayer(layers.Conv2DLayer):
    def __init__(self, incoming, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(inv_conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    #def get_W_shape(self):
    #    shape = super(Deconv2DLayer, self).get_W_shape()
    #    return (shape[1], shape[0]) + shape[2:]

    def convolve(self, input, **kwargs):
        shape = self.get_output_shape_for(input.shape)
        fake_output = T.alloc(0., *shape)
        border_mode = 'half' if self.pad == 'same' else self.pad
        
        w_shape = self.get_W_shape()
        w_shape = (w_shape[1], w_shape[0]) + w_shape[2:]
        shape = self.get_output_shape_for(self.input_layer.output_shape)
        W = self.W.transpose((1, 0, 2, 3))

        conved = self.convolution(fake_output, W,
                                  shape, w_shape,
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return theano.grad(None, wrt=fake_output, known_grads={conved: input})

class Deconv2DLayer_v2(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 W=lasagne.init.Orthogonal(),
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(W,
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)


if __name__ == '__main__':
    from lasagne.layers import *
    num_hid = 20
    num_in = 2
    x = InputLayer((None, num_in))
    inp = x

    hid = InputLayer((None, num_hid))
    hid_to_out = DenseLayer(hid, num_in)
    feat = DenseLayer(x, 256)

    def predict_input(xprev, hprev, prev):
        return xprev
    
    def predict_repr(x):
        return layers.get_output(feat, x)

    def predict_output(oprev, hcur):
        return layers.get_output(hid_to_out, hcur)
    x = FeedbackGRULayerClean(
        x,
        num_units=num_hid,
        predict_input=predict_input,
        predict_output=predict_output,
        predict_repr=predict_repr,
        repr_shape=(None, 256),
        out_shape=(None, num_in),
        n_steps=1)
    x = AddParams(x, [hid_to_out, feat])
    print(layers.get_all_params(x))
    fn = theano.function([inp.input_var], get_output(x, inp.input_var))
    x_example = np.ones((100, num_in)) * 100
    x_example = x_example.astype(np.float32)
    #print(fn(x_example)[:, -1])
