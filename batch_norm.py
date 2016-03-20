# -*- coding: utf-8 -*-

"""
Preliminary implementation of batch normalization for Lasagne.
Does not include a way to properly compute the normalization factors over the
full training set for testing, but can be used as a drop-in for training and
validation.

Author: Jan Schlüter
"""

import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.random import get_rng
from lasagne.layers import MergeLayer
from lasagne import nonlinearities, init

class BatchNormLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axes=None, epsilon=0.01, alpha=0.5,
            nonlinearity=None, additive_noise=None, **kwargs):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).

        @param incoming: `Layer` instance or expected input shape
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.additive_noise = additive_noise
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        dtype = theano.config.floatX
        self.mean = self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(lasagne.init.Constant(1), shape, 'std',
                                  trainable=False, regularizable=False)
        self.beta = self.add_param(lasagne.init.Constant(0), shape, 'beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(lasagne.init.Constant(1), shape, 'gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = input.mean(self.axes, keepdims=True)
            std = input.std(self.axes, keepdims=True)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * std)
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            std += 0 * running_std
        std += self.epsilon
        mean = T.addbroadcast(mean, *self.axes)
        std = T.addbroadcast(std, *self.axes)
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)

        if self.additive_noise is not None and deterministic is False:
            input = (input - mean) / std
            input = input + self._srng.normal(input.shape, avg=0.0, std=self.sigma)
            normalized = input * gamma + beta
        else:
            normalized = (input - mean) * (gamma / std) + beta
        return self.nonlinearity(normalized)

def batch_norm(layer, name=None):
    """
    Convenience function to apply batch normalization to a given layer's output.
    Will steal the layer's nonlinearity if there is one (effectively introducing
    the normalization right before the nonlinearity), and will remove the
    layer's bias if there is one (because it would be redundant).

    @param layer: The `Layer` instance to apply the normalization to; note that
        it will be irreversibly modified as specified above
    @return: A `BatchNormLayer` instance stacked on the given `layer`
    """
    if name is None:
        name = layer.name
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity, name=name)

class DecoderNormalizeLayer(lasagne.layers.MergeLayer):
    """
        Special purpose layer used to construct the ladder network
        See the ladder_network example.
    """
    def __init__(self, incoming, mean, std, epsilon=0.01, **kwargs):
        super(DecoderNormalizeLayer, self).__init__(
            [incoming, mean, std], **kwargs)
        self.epsilon = epsilon

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        input, mean, std = inputs
        return (input - mean) / (std + self.epsilon)


def FakeLayer(parent, value):

    def get_output_for(input, **kwargs):
        return input
    def get_params(**tags):
        return []
    value.input_layer = parent
    value.output_shape = parent.output_shape
    value.get_output_for = get_output_for
    value.get_params = get_params
    return value

class DenoiseLayer(MergeLayer):
    """
        Special purpose layer used to construct the ladder network
        See the ladder_network example.
    """
    def __init__(self, u_net, z_net,
                 nonlinearity=nonlinearities.sigmoid, **kwargs):
        super(DenoiseLayer, self).__init__([u_net, z_net], **kwargs)

        u_shp, z_shp = self.input_shapes


        if not u_shp[-1] == z_shp[-1]:
            raise ValueError("last dimension of u and z  must be equal"
                             " u was %s, z was %s" % (str(u_shp), str(z_shp)))
        self.num_inputs = z_shp[-1]
        self.nonlinearity = nonlinearity
        constant = init.Constant
        self.a1 = self.add_param(constant(0.), (self.num_inputs,), name="a1")
        self.a2 = self.add_param(constant(1.), (self.num_inputs,), name="a2")
        self.a3 = self.add_param(constant(0.), (self.num_inputs,), name="a3")
        self.a4 = self.add_param(constant(0.), (self.num_inputs,), name="a4")

        self.c1 = self.add_param(constant(0.), (self.num_inputs,), name="c1")
        self.c2 = self.add_param(constant(1.), (self.num_inputs,), name="c2")
        self.c3 = self.add_param(constant(0.), (self.num_inputs,), name="c3")

        self.c4 = self.add_param(constant(0.), (self.num_inputs,), name="c4")

        self.b1 = self.add_param(constant(0.), (self.num_inputs,),
                                 name="b1", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        output_shape = list(input_shapes[0])  # make a mutable copy
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        u, z_lat = inputs
        sigval = self.c1 + self.c2*z_lat
        sigval += self.c3*u + self.c4*z_lat*u
        sigval = self.nonlinearity(sigval)
        z_est = self.a1 + self.a2 * z_lat + self.b1*sigval
        z_est += self.a3*u + self.a4*z_lat*u

        return z_est

class NormalizeLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axes=None, epsilon=0.01, alpha='single_pass',
                 return_stats=False, stat_indices=None,
                 **kwargs):
        """
        This layer is a modified version of code originally written by
        Jan Schluter.
        Instantiates a layer performing batch normalization of its inputs [1]_.
        Params
        ------
        incoming: `Layer` instance or expected input shape
        axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
            If alpha is none we'll assume that the entire training set
            is passed through in one batch.
        return_stats: return mean and std
        stat_indices if slice object only calc stats for these indices. Used
            semisupervsed learning
        Notes
        -----
        This layer accepts the keyword collect=True when get_output is
        called. Before evaluation you should collect the batchnormalizatino
        statistics by running all you data through a function with
        collect=True and deterministic=True
        Then you can evaluate.
        References
        ----------
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization:
               Accelerating deep network training by reducing internal
               covariate shift."
               arXiv preprint arXiv:1502.03167 (2015).
        """
        super(NormalizeLayer, self).__init__(incoming, **kwargs)
        self.return_stats = return_stats
        self.stat_indices = stat_indices
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("NormalizeLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        self.mean = self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.var = self.add_param(lasagne.init.Constant(1), shape, 'var',
                                  trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, collect=False,
                       **kwargs):

        if collect:
            # use this batch's mean and var
            if self.stat_indices is None:
                mean = input.mean(self.axes, keepdims=True)
                var = input.var(self.axes, keepdims=True)
            else:
                mean = input[self.stat_indices].mean(self.axes, keepdims=True)
                var = input[self.stat_indices].var(self.axes, keepdims=True)
            # and update the stored mean and var:
            # we create (memory-aliased) clones of the stored mean and var
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_var = theano.clone(self.var, share_inputs=False)
            # set a default update for them

            if self.alpha is not 'single_pass':
                running_mean.default_update = (
                    (1 - self.alpha) * running_mean + self.alpha * mean)
                running_var.default_update = (
                    (1 - self.alpha) * running_var + self.alpha * var)
            else:
                print "Collecting using single pass..."
                # this is ugly figure out what can be safely removed...
                running_mean.default_update = (0 * running_mean + 1.0 * mean)
                running_var.default_update = (0 * running_var + 1.0 * var)

            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            var += 0 * running_var

        elif deterministic:
            # use stored mean and var
            mean = self.mean
            var = self.var
        else:
            # use this batch's mean and var
            mean = input.mean(self.axes, keepdims=True)
            var = input.var(self.axes, keepdims=True)

        mean = T.addbroadcast(mean, *self.axes)
        var = T.addbroadcast(var, *self.axes)
        normalized = (input - mean) / T.sqrt(var + self.epsilon)

        if self.return_stats:
            return [normalized, mean, var]
        else:
            return normalized


class ScaleAndShiftLayer(lasagne.layers.Layer):
    """
    This layer is a modified version of code originally written by
    Jan Schluter.
    Used with the NormalizeLayer to construct a batchnormalization layer.
    Params
    ------
    incoming: `Layer` instance or expected input shape
    axes: int or tuple of int denoting the axes to normalize over;
        defaults to all axes except for the second if omitted (this will
        do the correct thing for dense layers and convolutional layers)
    """

    def __init__(self, incoming, axes=None, beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1), **kwargs):
        super(ScaleAndShiftLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        self.beta = self.add_param(beta, shape, name='beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(gamma, shape, name='gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        return input*gamma + beta
