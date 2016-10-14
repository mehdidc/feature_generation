from lasagne import layers, init
from lasagnekit.easy import layers_from_list_to_dict
from lasagne.nonlinearities import (
        linear, sigmoid, rectify, very_leaky_rectify, softmax, tanh)
from lasagnekit.layers import Deconv2DLayer
from helpers import FeedbackGRULayer, TensorDenseLayer
from layers import FeedbackGRULayerClean, AddParams
from helpers import Deconv2DLayer as deconv2d
from helpers import correct_over_op, over_op, sum_op, max_op, thresh_op, normalized_over_op, mask_op, mask_smooth_op, sub_op, normalized_sum_op
from helpers import wta_spatial, wta_k_spatial, wta_lifetime, wta_channel, wta_channel_strided, wta_fc_lifetime, wta_fc_sparse, norm_maxmin, max_k_spatial
from helpers import Repeat
from helpers import BrushLayer, GenericBrushLayer
from helpers import GaussianSampleLayer, ExpressionLayerMulti, axify
from helpers import recurrent_accumulation
import theano.tensor as T
import numpy as np


from utils.batch_norm import (
    NormalizeLayer,
    ScaleAndShiftLayer,
    DecoderNormalizeLayer,
    DenoiseLayer,
    FakeLayer)
from lasagne.layers import batch_norm

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils.sparsemax_theano import sparsemax

def sparsemax_seq(x):
    x = T.cast(x, theano.config.floatX)
    orig_shape = x.shape
    x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
    x =  sparsemax(x)
    x = x.reshape(orig_shape)
    return x

def softmax_seq(x):
    x = T.cast(x, theano.config.floatX)
    orig_shape = x.shape
    x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
    x = softmax(x)
    x = x.reshape(orig_shape)
    return x

def is_max_seq(x):
    m = x.max(axis=2, keepdims=True)
    return T.eq(x, m)# / m

def is_max(x):
    m = x.max(axis=1, keepdims=True)
    return T.eq(x, m)

get_nonlinearity = dict(
    linear=linear,
    sigmoid=sigmoid,
    rectify=rectify,
    relu=rectify,
    very_leaky_rectify=very_leaky_rectify,
    softmax=softmax,
    tanh=tanh,
    msigmoid=lambda x:1 - sigmoid(x)
)

normalize_funcs = {
    'maxmin': norm_maxmin,
    'sigmoid': T.nnet.sigmoid,
    'none': lambda x: x}

reduce_funcs = {
    'sum': sum_op,
    'normalized_sum': normalized_sum_op,
    'over': over_op,
    'normalized_over': normalized_over_op,
    'max': max_op,
    'new': lambda prev, new: new+prev-prev,
    'prev': lambda prev, new: prev+new-new,
    'mask_op': mask_op,
    'mask_smooth_op': mask_smooth_op,
    'sub_op': sub_op
}

proba_funcs = {
    'softmax': T.nnet.softmax,
    'sparsemax': sparsemax,
    'is_max': is_max
}

recurrent_models = {
    'lstm': layers.LSTMLayer,
    'gru': layers.GRULayer,
    'rnn': layers.RecurrentLayer
}

sparsemax_ = axify(sparsemax)
softmax_ = axify(softmax)

def model1(nb_filters=64, w=32, h=32, c=1):
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad=2,
            name="conv")
    l_wta = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta")
    l_wta = l_conv
    l_unconv = layers.Conv2DLayer(
            l_wta,
            num_filters=c,
            filter_size=(11, 11),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad=5,
            name='unconv')

    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    return layers_from_list_to_dict([l_in, l_conv, l_wta, l_unconv, l_out])


def model2(nb_filters=64, w=32, h=32, c=1):

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(3, 3),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_wta = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta")
    l_unconv = layers.Conv2DLayer(
            l_wta,
            num_filters=c,
            filter_size=(11, 11),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] +  l_convs + [l_wta, l_unconv, l_out])

def model3(nb_filters=64, w=32, h=32, c=1):
    """
    is the same than convnet_simple_2 except it uses (5 , 5) then (5, 5) then (5, 5) which gives 28 - 5*3 + 3 = 16 then (13, 13) for decoding which gives 16+13-1=28
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_wta = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta")
    l_unconv = layers.Conv2DLayer(
            l_wta,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta, l_unconv, l_out])


def model4(nb_filters=64, w=32, h=32, c=1):
    """
    is the same than 3 except it does wta_spatial for each layer in the encoder
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_conv = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta3")
    l_convs.append(l_conv)
    l_unconv = layers.Conv2DLayer(
            l_conv,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_unconv, l_out])


def model5(nb_filters=64, w=32, h=32, c=1):
    """
    is the same than convnet_simple_3 but introduces wta_lifetime of 20% after wta_spatial
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_lifetime(0.2), name="wta_lifetime")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out])



def model6(nb_filters=64, w=32, h=32, c=1):
    """
     is the same than convnet_simple_3 except it does wta_channel after wta_spatial
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta_spatial")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel, name="wta_channel")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out])

def model7(nb_filters=64, w=32, h=32, c=1):
    """
    is the same than convnet_simple_3 except it does wta_channel_strided with stride=2 after wta_spatial
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta_spatial")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=2), name="wta_channel")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out])


def model8(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    is the same than convnet_simple_7 except it does wta_channel_strided with stride=4
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta_spatial")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out])


def model9(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    is a new kind of model inspired from ladder networks
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv1 = layers.Conv2DLayer(
             l_in,
             num_filters=nb_filters,
             filter_size=(9, 9),
             nonlinearity=rectify,
             W=init.GlorotUniform(),
             name="conv1")
    l_convs.append(l_conv1)
    l_conv2 = layers.Conv2DLayer(
             l_conv1,
             num_filters=nb_filters,
             filter_size=(9, 9),
             nonlinearity=rectify,
             W=init.GlorotUniform(),
             name="conv2")
    l_convs.append(l_conv2)
    l_conv3 = layers.Conv2DLayer(
             l_conv2,
             num_filters=nb_filters,
             filter_size=(9, 9),
             nonlinearity=rectify,
             W=init.GlorotUniform(),
             name="conv3")
    l_convs.append(l_conv3)

    def sparse(L):
        return layers.NonlinearityLayer(L, wta_spatial)
    l_unconvs = []
    l_unconv2_a = layers.Conv2DLayer(
              l_conv3,
              num_filters=nb_filters,
              filter_size=(9, 9),
              nonlinearity=rectify,
              W=init.GlorotUniform(),
              pad='full',
              name='unconv2_a')
    l_unconvs.append(l_unconv2_a)
    l_unconv2 = layers.ElemwiseSumLayer([l_unconv2_a, sparse(l_conv2)], name="unconv2")
    l_unconvs.append(l_unconv2)

    l_unconv1_a = layers.Conv2DLayer(
              l_unconv2,
              num_filters=nb_filters,
              filter_size=(9, 9),
              nonlinearity=rectify,
              W=init.GlorotUniform(),
              pad='full',
              name='unconv1_a')
    l_unconvs.append(l_unconv1_a)
    l_unconv1 = layers.ElemwiseSumLayer([l_unconv1_a, sparse(l_conv1)], name="unconv1")
    l_unconvs.append(l_unconv1)

    l_unconv0 = layers.Conv2DLayer(
             l_unconv1,
             num_filters=c,
             filter_size=(9, 9),
             nonlinearity=linear,
             W=init.GlorotUniform(),
             pad='full',
             name='unconv0')
    l_unconvs.append(l_unconv0)
    l_out = layers.NonlinearityLayer(
            l_unconv0,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + l_unconvs + [l_out])

def model10(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    is the same than convnet_simple_8 except it has deep decoder
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)

    l_unconvs = []
    l_unconv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full',
            name="unconv2")
    l_unconvs.append(l_unconv)
    l_unconv = layers.Conv2DLayer(
            l_unconv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full',
            name="unconv1")
    l_unconvs.append(l_unconv)
    l_wta1 = layers.NonlinearityLayer(l_unconv, wta_spatial, name="wta_spatial")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + l_unconvs + [l_wta1, l_wta2, l_unconv, l_out])


def model11(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like 10 but have sparsity in two deconv layers instead of one
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)

    l_unconvs = []
    l_unconv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full',
            name="unconv2")
    l_unconvs.append(l_unconv)
    l_unconv = layers.NonlinearityLayer(l_unconv, wta_spatial)
    l_unconv = layers.Conv2DLayer(
            l_unconv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full',
            name="unconv1")
    l_unconvs.append(l_unconv)
    l_wta1 = layers.NonlinearityLayer(l_unconv, wta_spatial, name="wta_spatial")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + l_unconvs + [l_wta1, l_wta2, l_unconv, l_out])


def model12(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    a new kind of architecture where we reconstruct in different scales :
    do a conv1 on the input do a deconv on conv1, do a conv1 on conv2 and do a deconv on conv2,
    do a conv3 on conv2 and deconv on conv3. deconv is done with sparsity.
    then merge all deconv1 deconv2 deconv3 using sum and use sigmoid.
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)

    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers) / 2
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(n))
        sparse_layers.append(l)
        return l

    l_out1 = layers.Conv2DLayer(
            sparse(l_convs[0]),
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            sparse(l_convs[1]),
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            sparse(l_convs[2]),
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseSumLayer(out_layers)
    l_out = layers.NonlinearityLayer(
            l_out,
            sigmoid, name="output")
    print(l_out.output_shape)
    all_layers = [l_in] + l_convs + sparse_layers + out_layers + [l_out]
    return layers_from_list_to_dict(all_layers)


def model13(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like 12 but use maximum instead of sum when merging deconv layers
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)

    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers) / 2
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(n))
        sparse_layers.append(l)
        return l

    l_out1 = layers.Conv2DLayer(
            sparse(l_convs[0]),
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            sparse(l_convs[1]),
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            sparse(l_convs[2]),
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.maximum)
    l_out = layers.NonlinearityLayer(
            l_out,
            sigmoid, name="output")
    print(l_out.output_shape)
    all_layers = [l_in] + l_convs + sparse_layers + out_layers + [l_out]
    return layers_from_list_to_dict(all_layers)

def model14(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like 12 but use multiplication instead of sum when merging deconv layers
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)

    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers) / 2
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(n))
        sparse_layers.append(l)
        return l

    l_out1 = layers.Conv2DLayer(
            sparse(l_convs[0]),
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            sparse(l_convs[1]),
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            sparse(l_convs[2]),
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.mul, name="output")
    print(l_out.output_shape)
    all_layers = [l_in] + l_convs + sparse_layers + out_layers + [l_out]
    return layers_from_list_to_dict(all_layers)


def model15(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like 14 but meant to be used for bigger images because it has bigger filters
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(9, 9),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(9, 9),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(11, 11),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)

    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers) / 2
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(n))
        sparse_layers.append(l)
        return l

    l_out1 = layers.Conv2DLayer(
            sparse(l_convs[0]),
            num_filters=c,
            filter_size=(11, 11),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            sparse(l_convs[1]),
            num_filters=c,
            filter_size=(21, 21),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            sparse(l_convs[2]),
            num_filters=c,
            filter_size=(31, 31),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.mul, name="output")
    print(l_out.output_shape)
    all_layers = [l_in] + l_convs + sparse_layers + out_layers + [l_out]
    return layers_from_list_to_dict(all_layers)


def model16(nb_filters=64, w=32, h=32, c=1, nb_filters_recons=64, sparsity=True):
    """
    like 8 but use 6 layers instead of 3 in the encoder
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv4")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv5")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters_recons,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv6")
    l_convs.append(l_conv)
    l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta_spatial")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(25, 25),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out])


def model17(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like 14 but filters are doubled for each layer
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters * 2,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters * 4,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)

    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers) / 2
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(n))
        sparse_layers.append(l)
        return l

    l_out1 = layers.Conv2DLayer(
            sparse(l_convs[0]),
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            sparse(l_convs[1]),
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            sparse(l_convs[2]),
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.mul, name="output")
    print(l_out.output_shape)
    all_layers = [l_in] + l_convs + sparse_layers + out_layers + [l_out]
    return layers_from_list_to_dict(all_layers)


def model18(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5):
    """
    stadard conv aa without any sparsity
    """
    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters,
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        l_convs.append(l_conv)
        print(l_conv.output_shape)

    l_unconv = l_conv
    l_unconvs = []
    for i in range(nb_layers):
        if i == nb_layers - 1:
            nonlin = sigmoid
            nb = c
            name = "output"
        else:
            nonlin = rectify
            nb = nb_filters
            name = "unconv{}".format(i)
        l_unconv = Deconv2DLayer(
                l_unconv,
                num_filters=nb,
                filter_size=(s, s),
                nonlinearity=nonlin,
                W=init.GlorotUniform(),
                stride=2,
                name=name)
        print(l_unconv.output_shape)
        l_unconvs.append(l_unconv)
    all_layers = [l_in] + l_convs + l_unconvs
    return layers_from_list_to_dict(all_layers)


def model19(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    like 18 but similar to ladder defined in 14
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters,
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                pad=(s-1)/2,
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=(2*(i+1), 2*(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.mul),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model20(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    like model 18 but with sparsity
    """
    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters,
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        l_convs.append(l_conv)
        print(l_conv.output_shape)

    l_unconv = l_conv
    l_unconvs = []
    for i in range(nb_layers):
        if i == nb_layers - 1:
            l_wta1 = layers.NonlinearityLayer(l_unconv, wta_spatial, name="wta_spatial")
            l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
            l_unconv = l_wta2
            nonlin = sigmoid
            nb = c
            name = "output"
        else:

            nonlin = rectify
            nb = nb_filters
            name = "unconv{}".format(i)
        l_unconv = Deconv2DLayer(
                l_unconv,
                num_filters=nb,
                filter_size=(s, s),
                nonlinearity=nonlin,
                W=init.GlorotUniform(),
                stride=2,
                name=name)
        print(l_unconv.output_shape)
        l_unconvs.append(l_unconv)
    all_layers = [l_in] + l_convs + [l_wta1, l_wta2] + l_unconvs
    return layers_from_list_to_dict(all_layers)


def model21(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    like 19 but not ladder : only sparsity at the end as in 8
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters,
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                pad=(s-1)/2,
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
    l_conv_sparse = sparse(l_conv)
    l_unconv = Deconv2DLayer(
            l_conv_sparse,
            num_filters=c,
            filter_size=(2*(i+1), 2*(i+1)),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            stride=2**(i+1),
            name="out".format(i))
    print("unconv:{}".format(l_unconv.output_shape))
    l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model22(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like model8 but nb filters are doubled in each layer
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters * 2,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters * 4,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta_spatial")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out])


def model23(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like 17 but additive and without channel sparsity
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters * 2,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters * 4,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)

    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        return l

    l_out1 = layers.Conv2DLayer(
            sparse(l_convs[0]),
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            sparse(l_convs[1]),
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            sparse(l_convs[2]),
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.mul, name="output")
    print(l_out.output_shape)
    all_layers = [l_in] + l_convs + sparse_layers + out_layers + [l_out]
    return layers_from_list_to_dict(all_layers)

def model24(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    like 19 but similar to ladder defined in 14
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters*(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.mul),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model25(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    model24 but with more sparsity (+ wta_channel)
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel,
                name="wta_channel_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters*(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.mul),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model26(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    model25 but with wta_lifetime for 20%
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_lifetime(0.2),
                name="wta_lifetime_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters*(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.mul),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model27(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    model26 but without contribution from first conv layer int reconstruction
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_lifetime(0.2),
                name="wta_lifetime_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters*(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        if i > 0:
            l_conv_sparse = sparse(l_conv)
            l_unconv = Deconv2DLayer(
                    l_conv_sparse,
                    num_filters=c,
                    filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                    nonlinearity=linear,
                    W=init.GlorotUniform(),
                    stride=2**(i+1),
                    name="out{}".format(i))
            print("unconv:{}".format(l_unconv.output_shape))
            l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.mul),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)

def model28(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    model26 but additive
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_lifetime(0.2),
                name="wta_lifetime_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters*(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.add),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model29(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    model25 but additive
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel,
                name="wta_channel_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters*(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.add),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)

def model30(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    model25 but additive but without wta_channel on last layer
    """
    sparse_layers = []

    def sparse(l, channel=True):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        if channel:
            sparse_layers.append(l)
            l = layers.NonlinearityLayer(
                    l, wta_channel,
                    name="wta_channel_{}".format(n))
            sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters*(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        if i == nb_layers - 1:
            channel = False
        else:
            channel = True
        l_conv_sparse = sparse(l_conv, channel=channel)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.add),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)

def model31(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    model8 without sparsity
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_unconv = layers.Conv2DLayer(
            l_conv,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_unconv, l_out])

def model32(nb_filters=64, w=32, h=32, c=1, sparsity=True, nb_classes=10):
    """
    predictive version of model8
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)


    l_pre_y = layers.DenseLayer(layers.ExpressionLayer(l_conv, lambda x:x.max(axis=(2, 3), keepdims=True), output_shape='auto'),
                                nb_classes, nonlinearity=linear, name="pre_y")
    l_y = layers.NonlinearityLayer(l_pre_y, nonlinearity=softmax, name="y")

    l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta_spatial")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out] + [l_pre_y, l_y])

def model33(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            spatial_k=8,
            sparsity=True):
    """
    like 24 but with wta_k_spatial
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_k_spatial(2),
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        if n == 0:
            l = layers.NonlinearityLayer(l, wta_channel_strided(stride=4))
        if n == 1:
            l = layers.NonlinearityLayer(l, wta_channel_strided(stride=2))
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters*(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.mul),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)

def model34(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    like 24 but nb of filters getting smaller as we increase depth
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters/(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
            layers.ElemwiseMergeLayer(l_unconvs, T.mul),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model35(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    like 24 but additive
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters/(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
        layers.ElemwiseMergeLayer(l_unconvs, T.add),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)

def model36(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    like 35 but wta_k_spatial
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_k_spatial(1),
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters/(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
        layers.ElemwiseMergeLayer(l_unconvs, T.add),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)

def model37(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    like 36 but with wta_channel
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_k_spatial(1),
                name="wta_spatial_{}".format(n/2))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
            l, wta_channel,
            name="wta_channel_{}".format(n/2))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters/(2**(i)),
                filter_size=(s, s),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
        layers.ElemwiseMergeLayer(l_unconvs, T.add),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)

def model38(w=32, h=32, c=1,
            nb_filters=64,
            size_filters=5,
            nb_layers=5,
            sparsity=True):
    """
    like 36 but with tanh behind as activation function
    """
    sparse_layers = []

    def sparse(l):
        n = len(sparse_layers)
        l = layers.NonlinearityLayer(
                l, wta_k_spatial(1),
                name="wta_spatial_{}".format(n))
        sparse_layers.append(l)
        return l

    s = size_filters
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    l_convs = []
    l_unconvs = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters/(2**(i)),
                filter_size=(s, s),
                nonlinearity=tanh,
                W=init.GlorotUniform(),
                stride=2,
                name="conv{}".format(i))
        print("conv:{}".format(l_conv.output_shape))
        l_convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        l_unconv = Deconv2DLayer(
                l_conv_sparse,
                num_filters=c,
                filter_size=((s-2)*(2**(i+1)-1)+2**(i+1), (s-2)*(2**(i+1)-1)+2**(i+1)),
                nonlinearity=linear,
                W=init.GlorotUniform(),
                stride=2**(i+1),
                name="out{}".format(i))
        print("unconv:{}".format(l_unconv.output_shape))
        l_unconvs.append(l_unconv)
    print(len(l_unconvs))
    l_out = layers.NonlinearityLayer(
        layers.ElemwiseMergeLayer(l_unconvs, T.add),
            sigmoid, name="output")
    all_layers = [l_in] + l_convs + sparse_layers + l_unconvs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model39(nb_filters=64, w=32, h=32, c=1, num_factors=10, sparsity=True, nb_classes=10):
    """
    predictive version of model8 + hidden factors
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    h1 = layers.DenseLayer(l_conv, num_factors, nonlinearity=linear, name="factor1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    h2 = layers.DenseLayer(l_conv, num_factors, nonlinearity=linear, name="factor2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    h3 = layers.DenseLayer(l_conv, num_factors, nonlinearity=linear, name="factor3")

    l_pre_y = layers.DenseLayer(layers.ExpressionLayer(l_conv, lambda x:x.max(axis=(2, 3), keepdims=True), output_shape='auto'),
                                nb_classes, nonlinearity=linear, name="pre_y")
    l_y = layers.NonlinearityLayer(l_pre_y, nonlinearity=softmax, name="y")
    hidfactors = layers.ConcatLayer([h1, h2, h3], axis=1, name="hidfactors")
    factors = layers.ConcatLayer([hidfactors, l_y], axis=1, name="factors")

    l_gate = layers.DenseLayer(factors, np.prod(l_conv.output_shape[1:]), nonlinearity=rectify, name="gate")
    l_gate = layers.ReshapeLayer(l_gate, ([0],) + l_conv.output_shape[1:], name="gate")
    #l_conv_gated = layers.ElemwiseMergeLayer([l_conv, l_gate], T.mul, name="conv3_gated")
    l_conv_gated = l_gate

    #l_wta1 = layers.NonlinearityLayer(l_conv_gated, wta_spatial, name="wta_spatial")
    #l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
    l_wta1 = l_conv_gated
    l_wta2 = l_conv_gated
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [h1, h2, h3, hidfactors, factors] + [l_gate, l_conv_gated, l_wta1, l_wta2, l_unconv, l_out] + [l_pre_y, l_y])

def model40(nb_filters=64, w=32, h=32, c=1, num_factors=50, sparsity=True, nb_classes=10):
    """
    simpler version of model39
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)

    h = layers.DenseLayer(l_conv, num_factors, nonlinearity=rectify, name="factor3")
    h = layers.DenseLayer(h, num_factors, nonlinearity=rectify, name="factor3")

    l_pre_y = layers.DenseLayer(layers.ExpressionLayer(l_conv, lambda x:x.max(axis=(2, 3), keepdims=True), output_shape='auto'),
                                nb_classes, nonlinearity=linear, name="pre_y")
    l_y = layers.NonlinearityLayer(l_pre_y, nonlinearity=softmax, name="y")
    hidfactors = layers.ConcatLayer([h], axis=1, name="hidfactors")
    factors = layers.ConcatLayer([hidfactors, l_y], axis=1, name="factors")

    l_gate = layers.DenseLayer(factors, np.prod(l_conv.output_shape[1:]), nonlinearity=rectify, name="gate")
    l_gate = layers.ReshapeLayer(l_gate, ([0],) + l_conv.output_shape[1:], name="gate")
    #l_conv_gated = layers.ElemwiseMergeLayer([l_conv, l_gate], T.mul, name="conv3_gated")
    l_conv_gated = l_gate

    #l_wta1 = layers.NonlinearityLayer(l_conv_gated, wta_spatial, name="wta_spatial")
    #l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
    l_wta1 = l_conv_gated
    l_wta2 = l_conv_gated
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [hidfactors, factors] + [l_gate, l_conv_gated, l_wta1, l_wta2, l_unconv, l_out] + [l_pre_y, l_y])

def model41(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    from "Generalized Denoising Auto-Encoders as Generative
    Models"
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = layers.DenseLayer(l_in, 2000, nonlinearity=tanh, name="hid")
    l_pre_out = layers.DenseLayer(l_hid, num_units=c*w*h, nonlinearity=linear, name="pre_output")
    l_out = layers.NonlinearityLayer(l_pre_out, sigmoid, name="output")
    l_out = layers.ReshapeLayer(l_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in, l_hid, l_pre_out, l_out])

def model42(w=32, h=32, c=1, nb_hidden=[1000, 500, 250, 250, 250], sigma=0.4, nb_filters=None):
    """
    ladder network with fully connected layers
    """

    nb_hidden = [c * w * h] + nb_hidden
    nb_layers = len(nb_hidden)

    l_in = layers.InputLayer((None, c, w, h), name="input")
    # Encoder noisy path
    x_noisy = layers.GaussianNoiseLayer(l_in, sigma=sigma)
    h = x_noisy
    encoder = []
    encoder_fc = []
    encoder_normlz = []
    encoder_scaleshift = []
    encoder_out = []
    for i in range(1, nb_layers):
        z = layers.DenseLayer(h, nb_hidden[i], nonlinearity=linear, name="noisy_enc_dense{}".format(i))
        encoder_fc.append(z)
        z = NormalizeLayer(z)
        encoder_normlz.append(z)
        z = layers.GaussianNoiseLayer(z, sigma=sigma)
        z.name = "noisy_enc{}".format(i)
        encoder.append(z)
        z = ScaleAndShiftLayer(z)
        encoder_scaleshift.append(z)
        z = layers.NonlinearityLayer(z, rectify)
        h = z
        encoder_out.append(h)
    encoder_clean = []
    # Encoder normal path
    h = l_in
    i = 1
    for fc, scaleshift in zip(encoder_fc, encoder_scaleshift):
        z = layers.DenseLayer(h, nb_hidden[i], W=fc.W, b=fc.b, nonlinearity=linear)
        z = NormalizeLayer(z)
        z.name = "enc{}".format(i)
        encoder_clean.append(z)
        z = ScaleAndShiftLayer(z, beta=scaleshift.beta, gamma=scaleshift.gamma)
        z = layers.NonlinearityLayer(z, rectify)
        h = z
        i += 1
    # decoder
    i -= 1
    decoder = []
    for lay in reversed([x_noisy] + encoder):
        if i == len(nb_hidden) - 1:
            u = NormalizeLayer(encoder_out[-1])
            u = ScaleAndShiftLayer(u)
        else:
            u = layers.DenseLayer(u, nb_hidden[i], nonlinearity=linear)
            u = NormalizeLayer(u)
            u = ScaleAndShiftLayer(u)
        if len(lay.output_shape) > 2:
            z = layers.ReshapeLayer(lay, ([0], np.prod(lay.output_shape[1:])))
        else:
            z = lay

        nonlin = sigmoid
        v = DenoiseLayer(u, z, nonlinearity=nonlin, name="dec{}".format(i))
        if i == 0:
            v = layers.NonlinearityLayer(v, sigmoid, name="dec{}".format(i))
        else:
            mean = encoder_normlz[i - 1].mean
            std = T.sqrt(encoder_normlz[i - 1].var)
            v = DecoderNormalizeLayer(
                v,
                mean=FakeLayer(encoder_normlz[i - 1], mean),
                std=FakeLayer(encoder_normlz[i - 1], std))
        v.name = "dec{}".format(i)
        decoder.append(v)
        u = v
        i -= 1
    output = layers.ReshapeLayer(u, ([0],) + l_in.output_shape[1:], name="output")
    return layers_from_list_to_dict([l_in] + encoder + encoder_clean + decoder + [output])

def model43(nb_filters=64, w=32, h=32, c=1):
    """
    Deconvolutional networks, Matthew D. Zeiler, Dilip Krishnan, Graham W. Taylor and Rob Fergus, 2010
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad=2,
            name="conv")
    l_wta = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta")
    l_wta = l_conv
    l_unconv = layers.Conv2DLayer(
            l_wta,
            num_filters=c,
            filter_size=(11, 11),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad=5,
            name='unconv')

    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    return layers_from_list_to_dict([l_in, l_conv, l_wta, l_unconv, l_out])

def model44(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    from Contractive Autencoders, 2010
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = layers.DenseLayer(l_in, 1000, nonlinearity=sigmoid, name="hid")
    l_pre_out = layers.DenseLayer(l_hid, num_units=c*w*h, nonlinearity=linear, W=l_hid.W.T,
                                  name="pre_output")
    l_out = layers.NonlinearityLayer(l_pre_out, sigmoid, name="output")
    l_out = layers.ReshapeLayer(l_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in, l_hid, l_pre_out, l_out])


def model45(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    from Zero-bias auto-encoders
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = layers.DenseLayer(l_in, 1000, nonlinearity=lambda v: (T.abs_(v) > 1)*v, b=None,
                              name="hid")
    l_pre_out = layers.DenseLayer(l_hid, num_units=c*w*h, nonlinearity=linear,
                                  W=l_hid.W.T,
                                  name="pre_output")
    #l_out = layers.NonlinearityLayer(l_pre_out, sigmoid, name="output")
    l_out = layers.ReshapeLayer(l_pre_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in, l_hid, l_pre_out, l_out])

def model46(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    Mean-covariance auto-encoders
    http://www.iro.umontreal.ca/~memisevr/pubs/rae.pdf
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hidx = layers.DenseLayer(layers.DropoutLayer(l_in, 0.5, rescale=False), 500,
                               nonlinearity=linear,
                               name="hidx")
    l_hidy = layers.DenseLayer(layers.DropoutLayer(l_in, 0.5, rescale=False), 500,
                               nonlinearity=linear,
                               name="hidy")
    l_hid = layers.ElemwiseMergeLayer([l_hidx, l_hidy], T.mul, name="hid")
    l_hid = layers.DenseLayer(
        l_hid,
        500,
        nonlinearity=rectify,
        name="hid")

    l_hidxr = l_hidx
    l_hidyr = layers.DenseLayer(l_hid, 500, nonlinearity=linear, name="hidyr")
    l_hidr = layers.ElemwiseMergeLayer([l_hidxr, l_hidyr], T.mul, name="hidr")
    l_pre_out = layers.DenseLayer(l_hidr,
                                  W=l_hidy.W.T,
                                  num_units=c*w*h,
                                  nonlinearity=linear,
                                  name="pre_output")
    l_out = layers.NonlinearityLayer(l_pre_out, sigmoid, name="output")
    l_out = layers.ReshapeLayer(l_pre_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in, l_hid, l_pre_out, l_out])

def model47(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    Simple Denoising AA model
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = layers.DenseLayer(l_in, 1000, nonlinearity=sigmoid, name="hid")
    l_pre_out = layers.DenseLayer(l_hid, num_units=c*w*h, nonlinearity=linear, name="pre_output")
    l_out = layers.NonlinearityLayer(l_pre_out, sigmoid, name="output")
    l_out = layers.ReshapeLayer(l_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in, l_hid, l_pre_out, l_out])

def model48(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    same than model8 but filters are multiplied by two in each layer
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters * 2,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters * 2 * 2,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv3")
    l_convs.append(l_conv)
    l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta_spatial")
    l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=4), name="wta_channel")
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out])


def model49(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    bigger version of model41
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid1 = layers.DenseLayer(l_in, 2000, nonlinearity=rectify, name="hid1")
    l_hid2 = layers.DenseLayer(l_hid1, 2000, nonlinearity=rectify, name="hid2")
    l_hid3 = layers.DenseLayer(l_hid2, 2000, nonlinearity=rectify, name="hid3")
    l_hid4 = layers.DenseLayer(l_hid3, 2000, nonlinearity=rectify, name="hid4")
    l_pre_out = layers.DenseLayer(l_hid4, num_units=c*w*h, nonlinearity=linear, name="pre_output")
    l_out = layers.NonlinearityLayer(l_pre_out, sigmoid, name="output")
    l_out = layers.ReshapeLayer(l_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in, l_hid1, l_pre_out, l_out])


def model50(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    Pyramidal auto-encoder!
    """
    l_in_scale1 = layers.InputLayer((None, c, w, h), name="scale1_input")
    l_in_scale2 = layers.InputLayer((None, c, w/2, h/2), name="scale2_input")
    l_in_scale3 = layers.InputLayer((None, c, w/4, h/4), name="scale3_input")
    l_in_scale4 = layers.InputLayer((None, c, w/8, h/8), name="scale4_input")
    inputs = [l_in_scale1, l_in_scale2, l_in_scale3, l_in_scale4]

    l_conv1 = layers.Conv2DLayer(
           l_in_scale1,
           num_filters=nb_filters,
           filter_size=(2, 2),
           stride=2,
           nonlinearity=rectify,
           W=init.GlorotUniform(),
           name='conv1')
    l_conv2 = layers.Conv2DLayer(
           l_conv1,
           num_filters=nb_filters,
           filter_size=(2, 2),
           stride=2,
           nonlinearity=rectify,
           W=init.GlorotUniform(),
           name='conv2')
    l_conv3 = layers.Conv2DLayer(
           l_conv2,
           num_filters=nb_filters,
           filter_size=(2, 2),
           stride=2,
           nonlinearity=rectify,
           W=init.GlorotUniform(),
           name='conv3')
    encoder = [l_conv1, l_conv2, l_conv3]
    l_scale4_output = layers.Conv2DLayer(
        l_conv3,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale4_output',
    )
    l_scale4_up = Deconv2DLayer(
        l_scale4_output,
        num_filters=c,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv4_up',
    )
    #l_scale3_prep = layers.ConcatLayer([l_in_scale3, l_scale4_up], axis=1)
    l_scale3_prep = l_scale4_up
    l_scale3_output = layers.Conv2DLayer(
        l_scale3_prep,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale3_output')

    l_scale3_up = Deconv2DLayer(
        l_scale3_output,
        num_filters=c,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv3_up',
    )
    #l_scale2_prep = layers.ConcatLayer([l_in_scale2, l_scale3_up], axis=1)
    l_scale2_prep = l_scale3_up
    l_scale2_output = layers.Conv2DLayer(
        l_scale2_prep,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale2_output')

    l_scale2_up = Deconv2DLayer(
        l_scale2_output,
        num_filters=c,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv2_up',
    )
    #l_scale1_prep = layers.ConcatLayer([l_in_scale1, l_scale2_up], axis=1)
    l_scale1_prep = l_scale2_up
    l_scale1_output = layers.Conv2DLayer(
        l_scale1_prep,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale1_output')
    outputs = [l_scale1_output, l_scale2_output, l_scale3_output, l_scale4_output]
    return layers_from_list_to_dict(inputs + encoder + outputs)

def model51(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    Pyramidal auto-encoder using hints in each scale
    """
    l_in_scale1 = layers.InputLayer((None, c, w, h), name="scale1_input")
    l_in_scale2 = layers.InputLayer((None, c, w/2, h/2), name="scale2_input")
    l_in_scale3 = layers.InputLayer((None, c, w/4, h/4), name="scale3_input")
    l_in_scale4 = layers.InputLayer((None, c, w/8, h/8), name="scale4_input")
    inputs = [l_in_scale1, l_in_scale2, l_in_scale3, l_in_scale4]

    l_conv1 = layers.Conv2DLayer(
           l_in_scale1,
           num_filters=nb_filters,
           filter_size=(2, 2),
           stride=2,
           nonlinearity=rectify,
           W=init.GlorotUniform(),
           name='conv1')
    l_conv2 = layers.Conv2DLayer(
           l_conv1,
           num_filters=nb_filters,
           filter_size=(2, 2),
           stride=2,
           nonlinearity=rectify,
           W=init.GlorotUniform(),
           name='conv2')
    l_conv3 = layers.Conv2DLayer(
           l_conv2,
           num_filters=nb_filters,
           filter_size=(2, 2),
           stride=2,
           nonlinearity=rectify,
           W=init.GlorotUniform(),
           name='conv3')
    encoder = [l_conv1, l_conv2, l_conv3]
    l_scale4_output = layers.Conv2DLayer(
        l_conv3,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale4_output',
    )
    l_scale4_up = Deconv2DLayer(
        l_scale4_output,
        num_filters=c,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv4_up',
    )
    l_scale3_prep = layers.ConcatLayer([l_in_scale3, l_scale4_up], axis=1)
    #l_scale3_prep = l_scale4_up
    l_scale3_output = layers.Conv2DLayer(
        l_scale3_prep,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale3_output')

    l_scale3_up = Deconv2DLayer(
        l_scale3_output,
        num_filters=c,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv3_up',
    )
    l_scale2_prep = layers.ConcatLayer([l_in_scale2, l_scale3_up], axis=1)
    #l_scale2_prep = l_scale3_up
    l_scale2_output = layers.Conv2DLayer(
        l_scale2_prep,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale2_output')

    l_scale2_up = Deconv2DLayer(
        l_scale2_output,
        num_filters=c,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv2_up',
    )
    l_scale1_prep = layers.ConcatLayer([l_in_scale1, l_scale2_up], axis=1)
    #l_scale1_prep = l_scale2_up
    l_scale1_output = layers.Conv2DLayer(
        l_scale1_prep,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale1_output')
    outputs = [l_scale1_output, l_scale2_output, l_scale3_output, l_scale4_output]
    return layers_from_list_to_dict(inputs + encoder + outputs)

def model52(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    Pyramidal auto-encoder using hints in each scale
    with noise
    """
    def sparse(l):
        l = layers.GaussianNoiseLayer(l, 0.2)
        return l

    l_in_scale1 = layers.InputLayer((None, c, w, h), name="scale1_input")
    l_in_scale2 = layers.InputLayer((None, c, w/2, h/2), name="scale2_input")
    l_in_scale3 = layers.InputLayer((None, c, w/4, h/4), name="scale3_input")
    l_in_scale4 = layers.InputLayer((None, c, w/8, h/8), name="scale4_input")
    inputs = [l_in_scale1, l_in_scale2, l_in_scale3, l_in_scale4]

    l_conv1 = layers.Conv2DLayer(
           l_in_scale1,
           num_filters=nb_filters,
           filter_size=(2, 2),
           stride=2,
           nonlinearity=rectify,
           W=init.GlorotUniform(),
           name='conv1')
    l_conv2 = layers.Conv2DLayer(
           l_conv1,
           num_filters=nb_filters,
           filter_size=(2, 2),
           stride=2,
           nonlinearity=rectify,
           W=init.GlorotUniform(),
           name='conv2')
    l_conv3 = layers.Conv2DLayer(
           l_conv2,
           num_filters=nb_filters,
           filter_size=(2, 2),
           stride=2,
           nonlinearity=rectify,
           W=init.GlorotUniform(),
           name='conv3')
    encoder = [l_conv1, l_conv2, l_conv3]
    l_scale4_output = layers.Conv2DLayer(
        l_conv3,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale4_output',
    )
    l_scale4_up = Deconv2DLayer(
        l_scale4_output,
        num_filters=c,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv4_up',
    )
    l_scale3_prep = layers.ConcatLayer([sparse(l_in_scale3), l_scale4_up], axis=1)
    #l_scale3_prep = l_scale4_up
    l_scale3_output = layers.Conv2DLayer(
        l_scale3_prep,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale3_output')

    l_scale3_up = Deconv2DLayer(
        l_scale3_output,
        num_filters=c,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv3_up',
    )
    l_scale2_prep = layers.ConcatLayer([sparse(l_in_scale2), l_scale3_up], axis=1)
    #l_scale2_prep = l_scale3_up
    l_scale2_output = layers.Conv2DLayer(
        l_scale2_prep,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale2_output')

    l_scale2_up = Deconv2DLayer(
        l_scale2_output,
        num_filters=c,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv2_up',
    )
    l_scale1_prep = layers.ConcatLayer([sparse(l_in_scale1), l_scale2_up], axis=1)
    #l_scale1_prep = l_scale2_up
    l_scale1_output = layers.Conv2DLayer(
        l_scale1_prep,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name='scale1_output')
    outputs = [l_scale1_output, l_scale2_output, l_scale3_output, l_scale4_output]
    return layers_from_list_to_dict(inputs + encoder + outputs)


def model53(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    Pyramidal auto-encoder
    """
    l_in = layers.InputLayer((None, c, w/8, h/8), name="input")
    l_deconv1 = Deconv2DLayer(
        l_in,
        num_filters=nb_filters,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv4_up',
    )
    l_deconv2 = Deconv2DLayer(
        l_deconv1,
        num_filters=nb_filters,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv4_up')

    l_deconv3 = Deconv2DLayer(
        l_deconv2,
        num_filters=nb_filters,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv4_up')
    l_output = layers.Conv2DLayer(
        l_deconv3,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name="output"
    )
    return layers_from_list_to_dict([l_in, l_deconv1, l_deconv2, l_deconv3, l_output])


def model54(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    Pyramidal auto-encoder
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_in_rescaled = layers.Pool2DLayer(l_in, (8, 8), mode='average_inc_pad')
    l_deconv1 = Deconv2DLayer(
        l_in_rescaled,
        num_filters=nb_filters,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv4_up',
    )
    l_deconv2 = Deconv2DLayer(
        l_deconv1,
        num_filters=nb_filters,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv4_up')

    l_deconv3 = Deconv2DLayer(
        l_deconv2,
        num_filters=nb_filters,
        filter_size=(2, 2),
        stride=2,
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='unconv4_up')
    l_output = layers.Conv2DLayer(
        l_deconv3,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name="output"
    )
    return layers_from_list_to_dict([l_in, l_deconv1, l_deconv2, l_deconv3, l_output])


def model55(nb_filters=64,  w=32, h=32, c=1,
            use_wta_channel=True,
            use_wta_spatial=True,
            nb_filters_mul=1,
            wta_channel_stride=2,
            nb_layers=3,
            filter_size=5):
    """
    model8 but parametrized
    """
    if type(nb_filters) == int:
        nb_filters = [nb_filters] * nb_layers
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = l_in
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters[i] * nb_filters_mul**i,
                filter_size=(filter_size, filter_size),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                name="conv{}".format(i + 1))
        l_convs.append(l_conv)

    if use_wta_spatial is True:
        l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta_spatial")
    else:
        l_wta1 = l_conv
    if use_wta_channel is True:
        l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=wta_channel_stride), name="wta_channel")
    else:
        l_wta2 = l_wta1

    w_out = l_conv.output_shape[2]
    w_remaining = w - w_out + 1
    print(w_remaining)
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(w_remaining, w_remaining),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out])


def model56(nb_filters=64, w=32, h=32, c=1,
            nb_layers=3,
            use_wta_lifetime=True,
            wta_lifetime_perc=0.1,
            nb_hidden_units=1000,
            out_nonlin='sigmoid'):

    """
    Generalization of "Generalized Denoising Auto-Encoders as Generative
    Models"
    """
    if type(nb_hidden_units) == int:
        nb_hidden_units = [nb_hidden_units] * nb_layers
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hids = []
    l_hid = l_in
    for i in range(nb_layers):
        l_hid = layers.DenseLayer(l_hid, nb_hidden_units[i], nonlinearity=rectify, name="hid{}".format(i + 1))
        hids.append(l_hid)
    if use_wta_lifetime is True:
        l_hid = layers.NonlinearityLayer(l_hid, wta_fc_lifetime(wta_lifetime_perc), name="hid{}sparse".format(i))
        hids.append(l_hid)
    l_pre_out = layers.DenseLayer(l_hid, num_units=c*w*h, nonlinearity=linear, name="pre_output")
    nonlin = {'sigmoid': sigmoid, 'tanh': tanh}[out_nonlin]
    l_out = layers.NonlinearityLayer(l_pre_out, nonlin, name="output")
    l_out = layers.ReshapeLayer(l_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + hids + [l_pre_out, l_out])

def model57(nb_filters=64, w=32, h=32, c=1,
            use_wta_lifetime=True,
            wta_lifetime_perc=0.1,
            nb_hidden_units=1000,
            tied=False,
            out_nonlin='sigmoid'):

    """
    Generalization of "Generalized Denoising Auto-Encoders as Generative
    Models"
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hids = []
    l_hid = layers.DenseLayer(
        l_in,
        nb_hidden_units,
        nonlinearity=sigmoid,
        name="hid")
    hids.append(l_hid)
    if use_wta_lifetime is True:
        l_hid = layers.NonlinearityLayer(l_hid, wta_fc_lifetime(wta_lifetime_perc), name="hid_sparse")
        hids.append(l_hid)
    if tied:
        W = hids[0].W.T
    else:
        W = init.GlorotUniform()
    l_pre_out = layers.DenseLayer(l_hid, num_units=c*w*h, W=W, nonlinearity=linear, name="pre_output")
    nonlin = {'sigmoid': sigmoid, 'tanh': tanh}[out_nonlin]
    l_out = layers.NonlinearityLayer(l_pre_out, nonlin, name="output")
    l_out = layers.ReshapeLayer(l_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + hids + [l_pre_out, l_out])


def model58(nb_filters=64, w=32, h=32, c=1, filter_size=3, nb_layers=3, sparsity=True):
    """
    Pyramidal auto-encoder input-input
    """
    if type(filter_size) == int:
        filter_size = [filter_size] * (nb_layers + 1)
    l_in = layers.InputLayer((None, c, w, h), name="input")
    #l_in_rescaled = layers.Pool2DLayer(l_in, (2, 2), mode='average_inc_pad')
    l_conv = l_in
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(filter_size[i], filter_size[i]),
            pad='same',
            nonlinearity=rectify,
            W=init.GlorotUniform(),
        )
    l_output = layers.Conv2DLayer(
        l_conv,
        num_filters=c,
        filter_size=(filter_size[-1], filter_size[-1]),
        pad='same',
        nonlinearity=linear,
        W=init.GlorotUniform(),
        name="output"
    )
    return layers_from_list_to_dict([l_in, l_output])


def model59(nb_filters=64,  w=32, h=32, c=1,
            use_wta_channel=True,
            use_wta_spatial=True,
            nb_filters_mul=1,
            wta_channel_stride=2,
            nb_layers=3,
            filter_size=5):
    """
    model55 but mode=same
    """
    if type(nb_filters) == int:
        nb_filters = [nb_filters] * nb_layers
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = l_in
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
                l_conv,
                num_filters=nb_filters[i] * nb_filters_mul**i,
                filter_size=(filter_size, filter_size),
                nonlinearity=rectify,
                pad='same',
                W=init.GlorotUniform(),
                name="conv{}".format(i + 1))
        l_convs.append(l_conv)

    if use_wta_spatial is True:
        l_wta1 = layers.NonlinearityLayer(l_conv, wta_spatial, name="wta_spatial")
    else:
        l_wta1 = l_conv
    if use_wta_channel is True:
        l_wta2 = layers.NonlinearityLayer(l_wta1, wta_channel_strided(stride=wta_channel_stride), name="wta_channel")
    else:
        l_wta2 = l_wta1

    #w_out = l_conv.output_shape[2]
    #w_remaining = w - w_out + 1
    #print(w_remaining)
    l_unconv = layers.Conv2DLayer(
            l_wta2,
            num_filters=c,
            filter_size=(filter_size, filter_size),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='same',
            name='unconv')
    l_out = layers.NonlinearityLayer(
            l_unconv,
            sigmoid, name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + l_convs + [l_wta1, l_wta2, l_unconv, l_out])


def model60(nb_filters=64, w=32, h=32, c=1, filter_size=3, nb_layers=3, block_size=1, sparsity=True, nonlinearity='rectify'):
    """
    Residual Pyramidal auto-encoder input-input
    """
    if type(filter_size) == int:
        filter_size = [filter_size] * (nb_layers + 1)

    nonlin = get_nonlinearity[nonlinearity]
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    prev = l_conv
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(filter_size[i], filter_size[i]),
            pad='same',
            nonlinearity=linear,
            W=init.GlorotUniform(),
        )
        if i == 0:
            prev = l_conv
        if i % block_size == 0 and i > 0:
            l_conv_ = l_conv
            l_conv = layers.NonlinearityLayer(layers.ElemwiseSumLayer([prev, l_conv]),
                                              nonlinearity=nonlin)
            prev = l_conv_
        else:
            l_conv = layers.NonlinearityLayer(l_conv, nonlin)
    l_output = layers.Conv2DLayer(
        l_conv,
        num_filters=c,
        filter_size=(filter_size[-1], filter_size[-1]),
        pad='same',
        nonlinearity=linear,
        W=init.GlorotUniform(),
        name="output")
    return layers_from_list_to_dict([l_in, l_output])

def model61(nb_filters=64, w=32, h=32, c=1, up=2, filter_size=3, nb_layers=3, block_size=1, sparsity=True, nonlinearity='rectify'):
    """
    Residual Pyramidal auto-encoder input Nx
    """
    if type(filter_size) == int:
        filter_size = [filter_size] * (nb_layers + 1)

    nonlin = get_nonlinearity[nonlinearity]
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    prev = l_conv
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(filter_size[i], filter_size[i]),
            pad='same',
            nonlinearity=linear,
            W=init.GlorotUniform(),
        )
        if i == 0:
            prev = l_conv
        if i % block_size == 0 and i > 0:
            l_conv_ = l_conv
            l_conv = layers.NonlinearityLayer(layers.ElemwiseSumLayer([prev, l_conv]),
                                              nonlinearity=nonlin)
            prev = l_conv_
        else:
            l_conv = layers.NonlinearityLayer(l_conv, nonlin)
    l_conv = Deconv2DLayer(
        l_conv,
        num_filters=nb_filters,
        filter_size=(filter_size[-1], filter_size[-1]),
        stride=up,
        nonlinearity=nonlin,
        W=init.GlorotUniform()
    )
    r = filter_size[-1] - up - 1
    l_output = layers.Conv2DLayer(
        l_conv,
        num_filters=c,
        filter_size=(r, r),
        nonlinearity=linear,
        W=init.GlorotUniform(),
        name="output")
    return layers_from_list_to_dict([l_in, l_output])

def model62(nb_filters=64, w=32, h=32, c=1, up=2, filter_size=3, nb_layers=3, sparsity=True):
    """
    Pyramidal auto-encoder input-input upscaler
    """
    if type(filter_size) == int:
        filter_size = [filter_size] * (nb_layers + 1)
    l_in = layers.InputLayer((None, c, w, h), name="input")
    #l_in_rescaled = layers.Pool2DLayer(l_in, (2, 2), mode='average_inc_pad')
    l_conv = l_in
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(filter_size[i], filter_size[i]),
            pad='same',
            nonlinearity=rectify,
            W=init.GlorotUniform(),
        )
    l_conv = Deconv2DLayer(
        l_conv,
        num_filters=nb_filters,
        filter_size=(filter_size[-1], filter_size[-1]),
        stride=up,
        nonlinearity=rectify,
        W=init.GlorotUniform()
    )
    r = filter_size[-1] - up + 1
    l_output = layers.Conv2DLayer(
        l_conv,
        num_filters=c,
        filter_size=(r, r),
        nonlinearity=linear,
        W=init.GlorotUniform(),
        name="output")
    return layers_from_list_to_dict([l_in, l_output])

def model63(nb_filters=64, w=32, h=32, c=1, up=2, filter_size=3, nb_layers=3, sparsity=True, use_batch_norm=False):
    """
    Pyramidal auto-encoder input-input upscaler but begins with upscaling
    """
    if type(filter_size) == int:
        filter_size = [filter_size] * (nb_layers + 1)
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = Deconv2DLayer(
        l_in,
        num_filters=nb_filters,
        filter_size=(filter_size[0], filter_size[0]),
        stride=up,
        nonlinearity=rectify,
        W=init.GlorotUniform()
    )
    if use_batch_norm:
        l_conv = batch_norm(l_conv)
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(filter_size[i + 1], filter_size[i + 1]),
            pad='same',
            nonlinearity=rectify,
            W=init.GlorotUniform(),
        )
        if use_batch_norm:
            l_conv = batch_norm(l_conv)

    l_output = layers.Conv2DLayer(
        l_conv,
        num_filters=c,
        filter_size=(filter_size[-1], filter_size[-1]),
        nonlinearity=linear,
        pad='same',
        W=init.GlorotUniform(),
        name="output")
    return layers_from_list_to_dict([l_in, l_output])


def model64(nb_filters=64, w=32, h=32, c=1,
            nb_layers=3,
            use_wta_sparse=True,
            wta_sparse_perc=0.1,
            nb_hidden_units=1000,
            out_nonlin='sigmoid',
            use_batch_norm=False):

    """
    Generalization of "Generalized Denoising Auto-Encoders as Generative
    Models"
    """
    if type(nb_hidden_units) == int:
        nb_hidden_units = [nb_hidden_units] * nb_layers
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hids = []
    l_hid = l_in
    for i in range(nb_layers):
        l_hid = layers.DenseLayer(l_hid, nb_hidden_units[i], nonlinearity=rectify, name="hid{}".format(i + 1))
        hids.append(l_hid)
        if use_batch_norm:
            l_hid = batch_norm(l_hid)
    if use_wta_sparse is True:
        l_hid = layers.NonlinearityLayer(l_hid, wta_fc_sparse(wta_sparse_perc), name="hid{}sparse".format(i))
        hids.append(l_hid)
    l_pre_out = layers.DenseLayer(l_hid, num_units=c*w*h, nonlinearity=linear, name="pre_output")
    nonlin = {'sigmoid': sigmoid, 'tanh': tanh}[out_nonlin]
    l_out = layers.NonlinearityLayer(l_pre_out, nonlin, name="output")
    l_out = layers.ReshapeLayer(l_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + hids + [l_pre_out, l_out])



def model65(nb_filters=64, filter_size=3, w=32, h=32, c=1, down=2,
            block_size=3, use_batch_norm=False, residual=False):
    """
    Pyramidal auto-encoder
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_in_rescaled = layers.Pool2DLayer(l_in, (down, down), mode='average_inc_pad')
    nb_layers = int(np.log2(down))
    l = l_in_rescaled

    if type(nb_filters) == int:
        nb_filters = [nb_filters] * nb_layers

    for i in range(nb_layers):
        nb_filters_cur = nb_filters[i]
        l = deconv2d(
            l,
            num_filters=nb_filters_cur,
            filter_size=filter_size,
            pad=(filter_size - 1) / 2,
            stride=2,
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name='deconv{}'.format(i),
        )
        if use_batch_norm:
            l = batch_norm(l)
        first = l
        for j in range(block_size):
            l = layers.Conv2DLayer(
                l,
                num_filters=nb_filters_cur,
                filter_size=filter_size,
                pad='same',
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                name='deconv{}_{}'.format(i, j)
            )
            if use_batch_norm:
                l = batch_norm(l)
            if residual:
                l = layers.ElemwiseSumLayer([first, l])

    l_output = layers.Conv2DLayer(
        l,
        num_filters=c,
        filter_size=(3, 3),
        pad=1,
        nonlinearity=sigmoid,
        W=init.GlorotUniform(),
        name="output"
    )
    return layers_from_list_to_dict([l_in, l_output])


def model66(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    my new vertebrate architecture idea, based on model14 :
            conv1       - conv2         -  conv3
            sparseconv1 - sparseconv2   -  sparseconv3
            out1        - conv2_back1   -  conv3_back1
                        - out2          -  conv3_back2
                                        -  out3
            out = mul(out1, out2, out3)
    next thing to do :
            - share the weights of (conv3_back2->out3), (conv2_back1->out2) and (sparseconv1->out1)
            - share the weights of (conv3_back1->conv3_back2) and (sparseconv2->conv2_back1)
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * 3
    sparse_layers = []

    def sparse(l):
        name = l.name
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(name))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(name))
        sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv1 = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_conv1_sparse = sparse(l_conv1)
    l_conv2 = layers.Conv2DLayer(
            l_conv1,
            num_filters=nb_filters[1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_conv2_sparse = sparse(l_conv2)
    l_conv3 = layers.Conv2DLayer(
             l_conv2,
             num_filters=nb_filters[2],
             filter_size=(5, 5),
             nonlinearity=rectify,
             W=init.GlorotUniform(),
             name="conv3")
    l_conv3_sparse = sparse(l_conv3)

    l_conv3_back = l_conv3_sparse
    for i in range(2):
        l_conv3_back = layers.Conv2DLayer(
            l_conv3_back,
            num_filters=nb_filters[2 - i - 1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full'
        )

    l_conv2_back = l_conv2_sparse
    for i in range(1):
        l_conv2_back = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full'
        )
    l_conv1_back = l_conv1_sparse
    l_out1 = layers.Conv2DLayer(
            l_conv1_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            l_conv3_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=sigmoid,
            W=init.GlorotUniform(),
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.mul, name="output")
    all_layers = [l_in, l_conv1, l_conv2, l_conv3] + sparse_layers + [l_out1, l_out2, l_out3, l_out]
    return layers_from_list_to_dict(all_layers)


def model67(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like model66 vertebrate but with sum instead of multiplication
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * 3
    sparse_layers = []

    def sparse(l):
        name = l.name
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(name))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(name))
        sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv1 = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_conv1_sparse = sparse(l_conv1)
    l_conv2 = layers.Conv2DLayer(
            l_conv1,
            num_filters=nb_filters[1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_conv2_sparse = sparse(l_conv2)
    l_conv3 = layers.Conv2DLayer(
             l_conv2,
             num_filters=nb_filters[2],
             filter_size=(5, 5),
             nonlinearity=rectify,
             W=init.GlorotUniform(),
             name="conv3")
    l_conv3_sparse = sparse(l_conv3)

    l_conv3_back = l_conv3_sparse
    for i in range(2):
        l_conv3_back = layers.Conv2DLayer(
            l_conv3_back,
            num_filters=nb_filters[2 - i - 1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full'
        )

    l_conv2_back = l_conv2_sparse
    for i in range(1):
        l_conv2_back = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full'
        )
    l_conv1_back = l_conv1_sparse
    l_out1 = layers.Conv2DLayer(
            l_conv1_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            l_conv3_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.add)
    l_out = layers.NonlinearityLayer(l_out, sigmoid, name='output')
    all_layers = [l_in, l_conv1, l_conv2, l_conv3] + sparse_layers + [l_out1, l_out2, l_out3, l_out]
    return layers_from_list_to_dict(all_layers)


def model68(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like model67 verteberate but with weight sharing in the last layers
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * 3
    sparse_layers = []

    def sparse(l):
        name = l.name
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(name))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(name))
        sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv1 = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_conv1_sparse = sparse(l_conv1)
    l_conv2 = layers.Conv2DLayer(
            l_conv1,
            num_filters=nb_filters[1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_conv2_sparse = sparse(l_conv2)
    l_conv3 = layers.Conv2DLayer(
             l_conv2,
             num_filters=nb_filters[2],
             filter_size=(5, 5),
             nonlinearity=rectify,
             W=init.GlorotUniform(),
             name="conv3")
    l_conv3_sparse = sparse(l_conv3)

    l_conv3_back = l_conv3_sparse
    for i in range(2):
        l_conv3_back = layers.Conv2DLayer(
            l_conv3_back,
            num_filters=nb_filters[2 - i - 1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full'
        )

    l_conv2_back = l_conv2_sparse
    for i in range(1):
        l_conv2_back = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full'
        )
    l_conv1_back = l_conv1_sparse
    l_out1 = layers.Conv2DLayer(
            l_conv1_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=l_out1.W,
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            l_conv3_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=l_out1.W,
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.add)
    l_out = layers.NonlinearityLayer(l_out, sigmoid, name='output')
    all_layers = [l_in, l_conv1, l_conv2, l_conv3] + sparse_layers + [l_out1, l_out2, l_out3, l_out]
    return layers_from_list_to_dict(all_layers)

def model69(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    like model68 vertebrate but only two layers
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * 3
    sparse_layers = []

    def sparse(l):
        name = l.name
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(name))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(name))
        sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv1 = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_conv1_sparse = sparse(l_conv1)
    l_conv2 = layers.Conv2DLayer(
            l_conv1,
            num_filters=nb_filters[1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_conv2_sparse = sparse(l_conv2)
    l_conv2_back = l_conv2_sparse
    for i in range(1):
        l_conv2_back = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full'
        )
    l_conv1_back = l_conv1_sparse
    l_out1 = layers.Conv2DLayer(
            l_conv1_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=l_out1.W,
            pad='full',
            name='out2')
    out_layers = [l_out1, l_out2]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.add)
    l_out = layers.NonlinearityLayer(l_out, sigmoid, name='output')
    all_layers = [l_in, l_conv1, l_conv2] + sparse_layers + [l_out1, l_out2, l_out]
    return layers_from_list_to_dict(all_layers)


def model70(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    model68 vertebarte with stride (used for large images)
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * 3
    sparse_layers = []
    def sparse(l):
        name = l.name
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(name))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(name))
        sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv1 = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2,
            name="conv1")
    l_conv1_sparse = sparse(l_conv1)
    l_conv2 = layers.Conv2DLayer(
            l_conv1,
            num_filters=nb_filters[1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2,
            name="conv2")
    l_conv2_sparse = sparse(l_conv2)
    l_conv3 = layers.Conv2DLayer(
             l_conv2,
             num_filters=nb_filters[2],
             filter_size=(5, 5),
             nonlinearity=rectify,
             W=init.GlorotUniform(),
             stride=2,
             pad=(5-1)/2,
             name="conv3")
    l_conv3_sparse = sparse(l_conv3)

    l_conv3_back = l_conv3_sparse
    for i in range(2):
        l_conv3_back = deconv2d(
            l_conv3_back,
            num_filters=nb_filters[2 - i - 1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2
        )

    l_conv2_back = l_conv2_sparse
    for i in range(1):
        l_conv2_back = deconv2d(
            l_conv2_back,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2
        )
    l_conv1_back = l_conv1_sparse
    l_out1 = deconv2d(
            l_conv1_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2,
            name='out1')
    l_out2 = deconv2d(
            l_conv2_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2,
            name='out2')
    l_out3 = deconv2d(
            l_conv3_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2,
            name='out3')
    print(l_out1.output_shape, l_out2.output_shape, l_out3.output_shape)
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.add)
    l_out = layers.NonlinearityLayer(l_out, sigmoid, name='output')
    all_layers = [l_in, l_conv1, l_conv2, l_conv3] + sparse_layers + [l_out1, l_out2, l_out3, l_out]
    return layers_from_list_to_dict(all_layers)

def model71(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    model70 vertebrate with sharing
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * 3
    sparse_layers = []
    def sparse(l):
        name = l.name
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(name))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(name))
        sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv1 = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2,
            name="conv1")
    l_conv1_sparse = sparse(l_conv1)
    l_conv2 = layers.Conv2DLayer(
            l_conv1,
            num_filters=nb_filters[1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2,
            name="conv2")
    l_conv2_sparse = sparse(l_conv2)
    l_conv3 = layers.Conv2DLayer(
             l_conv2,
             num_filters=nb_filters[2],
             filter_size=(5, 5),
             nonlinearity=rectify,
             W=init.GlorotUniform(),
             stride=2,
             pad=(5-1)/2,
             name="conv3")
    l_conv3_sparse = sparse(l_conv3)

    l_conv3_back = l_conv3_sparse
    for i in range(2):
        l_conv3_back = deconv2d(
            l_conv3_back,
            num_filters=nb_filters[2 - i - 1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2
        )

    l_conv2_back = l_conv2_sparse
    for i in range(1):
        l_conv2_back = deconv2d(
            l_conv2_back,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2
        )
    l_conv1_back = l_conv1_sparse
    l_out1 = deconv2d(
            l_conv1_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            stride=2,
            pad=(5-1)/2,
            name='out1')
    l_out2 = deconv2d(
            l_conv2_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=l_out1.W,
            stride=2,
            pad=(5-1)/2,
            name='out2')
    l_out3 = deconv2d(
            l_conv3_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=l_out1.W,
            stride=2,
            pad=(5-1)/2,
            name='out3')
    print(l_out1.output_shape, l_out2.output_shape, l_out3.output_shape)
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.add)
    l_out = layers.NonlinearityLayer(l_out, sigmoid, name='output')
    all_layers = [l_in, l_conv1, l_conv2, l_conv3] + sparse_layers + [l_out1, l_out2, l_out3, l_out]
    return layers_from_list_to_dict(all_layers)


def model72(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    Discritized brush stroke
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * 3

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = layers.Conv2DLayer(
        l_in,
        num_filters=nb_filters[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad=2,
        name="conv1")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
        l_conv,
        num_filters=nb_filters[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad=2,
        name="conv2")
    l_convs.append(l_conv)
    l_conv = layers.Conv2DLayer(
        l_conv,
        num_filters=nb_filters[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad=2,
        name="conv3")
    l_convs.append(l_conv)
    l_wta_spatial = layers.NonlinearityLayer(
        l_conv,
        wta_spatial,
        name="wta_spatial")
    size = 3
    W = np.zeros((nb_filters[2], nb_filters[2], c, size, size))
    i = np.arange(nb_filters[2])
    W[i, i, :, :, :] = 1./(size*size)
    W = W.reshape((nb_filters[2], nb_filters[2] * c, size, size))
    W = W.astype(np.float32)
    print(W[0, 0])
    print(W[0, 1])
    l_unconv = l_wta_spatial
    l_unconv = layers.Conv2DLayer(
        l_unconv,
        num_filters=nb_filters[2] * c,
        filter_size=(size, size),
        nonlinearity=linear,
        W = W,  # square brush
        b=init.Constant(0),
        pad=(size - 1) / 2,
        name='unconv')
    l_unconv.params[l_unconv.W].remove('trainable')
    l_unconv.params[l_unconv.b].remove('trainable')
    l_wta_channel = layers.NonlinearityLayer(
        l_unconv,
        wta_channel,
        name='wta_channel')

    def fn(x):
        x = x.reshape((x.shape[0], c, nb_filters[2], x.shape[2], x.shape[3]))
        return x.sum(axis=2)
    l_out = layers.ExpressionLayer(
        l_wta_channel,
        fn
    )
    l_out = layers.NonlinearityLayer(
        l_out,
        linear, name="output")
    all_layers = (
        [l_in] +
        l_convs +
        [l_unconv, l_wta_spatial, l_wta_channel, l_out])
    return layers_from_list_to_dict(all_layers)

def model73(nb_filters=64, w=32, h=32, c=1,
            nb_layers=3,
            filter_size=5,
            use_channel=True,
            use_spatial=True,
            spatial_k=1,
            channel_stride=2,
            weight_sharing=False,
            merge_op='sum'):
    """
    parametrized version of vertebrate model67
    """
    merge_op = {'sum': T.add, 'mul': T.mul, 'over': over_op}[merge_op]
    if type(filter_size) != list:
        filter_size = [filter_size] * nb_layers
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * nb_layers
    if type(channel_stride) != list:
        channel_stride = [channel_stride] * nb_layers
    if type(spatial_k) != list:
        spatial_k = [spatial_k] * nb_layers

    if type(weight_sharing) != list:
        weight_sharing = [weight_sharing] * nb_layers
    sparse_layers = []


    def sparse(l):
        name = l.name
        idx = int(name.replace('conv', '')) - 1
        if use_spatial:
            l = layers.NonlinearityLayer(
                l, (wta_spatial
                    if spatial_k[idx] == 1
                    else wta_k_spatial(spatial_k[idx])),
                name="wta_spatial_{}".format(name))
            sparse_layers.append(l)
        if use_channel:
            l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=channel_stride[idx]),
                name="wta_channel_{}".format(name))
            sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    convs = []
    convs_sparse = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters[i],
            filter_size=(filter_size[i], filter_size[i]),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv{}".format(i + 1))
        convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        convs_sparse.append(l_conv_sparse)

    conv_backs = []
    back = {}
    back_layers = {}
    for i in range(nb_layers): #[0, 1, 2]
        l_conv_back = convs_sparse[i]
        for j in range(i): # for 0 : [], for 1 : [0], for 2 : [0, 1]
            if weight_sharing[i - j - 1] and i > 0 and j > 0:
                W = back_layers[(i - 1, j - 1)].W
            else:
                W = init.GlorotUniform()
            l_conv_back = layers.Conv2DLayer(
                l_conv_back,
                num_filters=nb_filters[i - j - 1],
                filter_size=(filter_size[i - j - 1], filter_size[i - j - 1]),
                nonlinearity=rectify,
                W=W,
                pad='full',
                name='conv_back_{}_{}'.format(i + 1, j + 1)
            )
            back_layers[(i, j)] = l_conv_back
            #back[(i, j)] = l_conv_back.W
        #l_conv_back.name = 'conv_back{}'.format(i + 1)
        conv_backs.append(l_conv_back)
    #print(conv_backs)
    outs = []
    for i, conv_back in enumerate(conv_backs):
        if i == 0 or not weight_sharing[0]:
            W = init.GlorotUniform()
        else:
            W = outs[0].W
        l_out = layers.Conv2DLayer(
            conv_back,
            num_filters=c,
            filter_size=(filter_size[0], filter_size[0]),
            nonlinearity=linear,
            W=W,
            pad='full',
            name='out{}'.format(i + 1))
        outs.append(l_out)
    l_out = layers.ElemwiseMergeLayer(outs, merge_op)
    l_out = layers.NonlinearityLayer(l_out, sigmoid, name='output')
    all_layers = [l_in] + convs + sparse_layers + back_layers.values() + outs + [l_out]
    return layers_from_list_to_dict(all_layers)

def model74(nb_filters=64, w=32, h=32, c=1,
            nb_layers=3,
            filter_size=5,
            use_channel=True,
            use_spatial=True,
            spatial_k=1,
            channel_stride=2,
            weight_sharing=False,
            merge_op='sum'):
    """
    model73 but with conv pad=same so that we are not limited by nb of layers
    """
    merge_op = {'sum': T.add, 'mul': T.mul}[merge_op]
    if type(filter_size) != list:
        filter_size = [filter_size] * nb_layers
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * nb_layers
    if type(channel_stride) != list:
        channel_stride = [channel_stride] * nb_layers
    if type(spatial_k) != list:
        spatial_k = [spatial_k] * nb_layers
    if type(weight_sharing) != list:
        weight_sharing = [weight_sharing] * nb_layers
    sparse_layers = []
    print('nb_filters : {}'.format(nb_filters))

    def sparse(l):
        name = l.name
        idx = int(name.replace('conv', '')) - 1
        if use_spatial:
            l = layers.NonlinearityLayer(
                l, (wta_spatial
                    if spatial_k[idx] == 1
                    else wta_k_spatial(spatial_k[idx])),
                name="wta_spatial_{}".format(name))
            sparse_layers.append(l)
        if use_channel:
            l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=channel_stride[idx]),
                name="wta_channel_{}".format(name))
            sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    convs = []
    convs_sparse = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters[i],
            filter_size=(filter_size[i], filter_size[i]),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='same',
            name="conv{}".format(i + 1))
        convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        convs_sparse.append(l_conv_sparse)

    conv_backs = []
    back = {}
    for i in range(nb_layers): #[0, 1, 2]
        l_conv_back = convs_sparse[i]
        for j in range(i): # for 0 : [], for 1 : [0], for 2 : [0, 1]
            if weight_sharing[i - j - 1] and i > 0 and j > 0:
                W = back[(i - 1, j - 1)]
            else:
                W = init.GlorotUniform()
            l_conv_back = layers.Conv2DLayer(
                l_conv_back,
                num_filters=nb_filters[i - j - 1],
                filter_size=(filter_size[i - j - 1], filter_size[i - j - 1]),
                nonlinearity=rectify,
                W=W,
                pad='same'
            )
            back[(i, j)] = l_conv_back.W
        l_conv_back.name = 'conv_back{}'.format(i + 1)
        conv_backs.append(l_conv_back)

    outs = []
    print(weight_sharing)
    for i, conv_back in enumerate(conv_backs):
        if i == 0 or not weight_sharing[0]:
            W = init.GlorotUniform()
        else:
            W = outs[0].W
        l_out = layers.Conv2DLayer(
            conv_back,
            num_filters=c,
            filter_size=(filter_size[0], filter_size[0]),
            nonlinearity=linear,
            W=W,
            pad='same',
            name='out{}'.format(i + 1))
        outs.append(l_out)
    l_out = layers.ElemwiseMergeLayer(outs, merge_op)
    l_out = layers.NonlinearityLayer(l_out, sigmoid, name='output')
    all_layers = [l_in] + convs + sparse_layers + conv_backs + outs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model75(w=32, h=32, c=1,
            nb_layers=3,
            nb_units=1000,
            n_steps=10,
            patch_size=3,
            nonlin='rectify'):
    """
    Simple continuous brush stroke without recurrence
    """
    def init_method():
        return init.GlorotUniform(gain='relu')
    if type(nb_units) != list:
        nb_units = [nb_units] * nb_layers
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = l_in
    nonlin = get_nonlinearity[nonlin]
    hids = []
    for i in range(nb_layers):
        l_hid = layers.DenseLayer(
            l_hid, nb_units[i],
            W=init_method(),
            nonlinearity=nonlin,
            name="hid{}".format(i + 1))
        #l_hid = batch_norm(l_hid)
        hids.append(l_hid)
    l_coord = layers.DenseLayer(
        l_hid,
        n_steps * 5,
        nonlinearity=linear,
        W=init.GlorotUniform(),
        name="coord")
    #l_coord = batch_norm(l_coord)
    l_hid = layers.ReshapeLayer(l_coord, ([0], n_steps, 5), name="hid3")
    l_brush = BrushLayer(
        l_hid,
        w, h,
        n_steps=n_steps,
        patch=np.ones((patch_size, patch_size)),
        name="brush")
    l_out = layers.ReshapeLayer(l_brush, ([0], c, w, h), name="output")
    l_out = layers.NonlinearityLayer(
        l_out,
        nonlinearity=linear,
        name="output")
    return layers_from_list_to_dict([l_in]+ hids + [l_coord, l_brush, l_out])


def model76(c=1, w=28, h=28, nb_layers=3, nb_filters=128, filter_size=5, patch_size=5, nonlin='rectify'):
    """
    discritized brush stroke parametrized version
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * nb_layers
    if type(filter_size) != list:
        filter_size = [filter_size] * nb_layers
    nonlin = get_nonlinearity[nonlin]

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_convs = []
    l_conv = l_in
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters[i],
            filter_size=(filter_size[i], filter_size[i]),
            nonlinearity=nonlin,
            W=init.GlorotUniform(),
            pad=(filter_size[i] - 1) / 2,
            name="conv{}".format(i + 1))
        l_convs.append(l_conv)
    l_wta_spatial = layers.NonlinearityLayer(
        l_conv,
        wta_spatial,
        name="wta_spatial")
    size = patch_size
    W = np.zeros((nb_filters[-1], nb_filters[-1], c, size, size))
    i = np.arange(nb_filters[-1])
    W[i, i, :, :, :] = 1./(size*size)
    W = W.reshape((nb_filters[-1], nb_filters[-1] * c, size, size))
    W = W.astype(np.float32)
    l_unconv = l_wta_spatial
    l_unconv = layers.Conv2DLayer(
        l_unconv,
        num_filters=nb_filters[-1] * c,
        filter_size=(size, size),
        nonlinearity=linear,
        W = W,  # square brush
        b=init.Constant(0),
        pad=(size - 1) / 2,
        name='unconv')
    print(l_unconv.output_shape)
    l_unconv.params[l_unconv.W].remove('trainable')
    l_unconv.params[l_unconv.b].remove('trainable')
    l_wta_channel = layers.NonlinearityLayer(
        l_unconv,
        wta_channel,
        name='wta_channel')

    def fn(x):
        x = x.reshape((x.shape[0], c, nb_filters[-1], x.shape[2], x.shape[3]))
        return x.sum(axis=2)
    l_out = layers.ExpressionLayer(
        l_wta_channel,
        fn
    )
    l_out = layers.NonlinearityLayer(
        l_out,
        linear, name="output")
    all_layers = (
        [l_in] +
        l_convs +
        [l_unconv, l_wta_spatial, l_wta_channel, l_out])
    return layers_from_list_to_dict(all_layers)


def model77(w=32, h=32, c=1,
            nb_fc_layers=3,
            nb_recurrent_layers=1,
            nb_recurrent_units=100,
            nb_fc_units=1000,
            n_steps=10,
            patch_size=3,
            nonlin='rectify'):
    """
    Simple continuous brush stroke with recurrence
    """
    def init_method():
        return init.GlorotUniform(gain='relu')
    if type(nb_fc_units) != list:
        nb_fc_units = [nb_fc_units] * nb_fc_layers
    if type(nb_recurrent_units) != list:
        nb_recurrent_units = [nb_recurrent_units] * nb_recurrent_layers
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = l_in
    nonlin = get_nonlinearity[nonlin]
    hids = []
    for i in range(nb_fc_layers):
        l_hid = layers.DenseLayer(
            l_hid, nb_fc_units[i],
            W=init_method(),
            nonlinearity=nonlin,
            name="hid{}".format(i + 1))
        hids.append(l_hid)
    l_hid = Repeat(l_hid, n_steps)
    for i in range(nb_recurrent_layers):
        l_hid = layers.GRULayer(l_hid, nb_recurrent_units[i])
    l_coord = layers.GRULayer(l_hid, 5, name="coord")
    l_hid = layers.ReshapeLayer(l_coord, ([0], n_steps, 5), name="hid3")
    l_brush = BrushLayer(
        l_hid,
        w, h,
        n_steps=n_steps,
        patch=np.ones((patch_size, patch_size)),
        name="brush")
    l_out = layers.ReshapeLayer(l_brush, ([0], c, w, h), name="output")
    l_out = layers.NonlinearityLayer(
        l_out,
        nonlinearity=linear,
        name="output")
    all_layers = [l_in] + hids + [l_coord, l_brush, l_out]
    return layers_from_list_to_dict(all_layers)


def model78(w=32, h=32, c=1,
            nb_fc_layers=3,
            nb_recurrent_layers=1,
            nb_recurrent_units=100,
            nb_fc_units=1000,
            n_steps=10,
            patch_size=3,
            nonlin='rectify'):
    """
    model77 but with sigmoid
    """
    def init_method():
        return init.GlorotUniform(gain='relu')
    if type(nb_fc_units) != list:
        nb_fc_units = [nb_fc_units] * nb_fc_layers
    if type(nb_recurrent_units) != list:
        nb_recurrent_units = [nb_recurrent_units] * nb_recurrent_layers
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = l_in
    nonlin = get_nonlinearity[nonlin]
    hids = []
    for i in range(nb_fc_layers):
        l_hid = layers.DenseLayer(
            l_hid, nb_fc_units[i],
            W=init_method(),
            nonlinearity=nonlin,
            name="hid{}".format(i + 1))
        hids.append(l_hid)
    l_hid = Repeat(l_hid, n_steps)
    for i in range(nb_recurrent_layers):
        l_hid = layers.GRULayer(l_hid, nb_recurrent_units[i])
    l_coord = layers.GRULayer(l_hid, 5, name="coord")
    l_hid = layers.ReshapeLayer(l_coord, ([0], n_steps, 5), name="hid3")
    l_brush = BrushLayer(
        l_hid,
        w, h,
        n_steps=n_steps,
        patch=np.ones((patch_size, patch_size)),
        name="brush")
    l_out = layers.ReshapeLayer(l_brush, ([0], c, w, h), name="output")
    l_out = layers.BiasLayer(l_out, b=init.Constant(-1.)) # because we are assuming the prev layer is between 0 and 1, we 'center' it at the beginning
    l_out = layers.NonlinearityLayer(
        l_out,
        nonlinearity=sigmoid,
        name="output")
    all_layers = [l_in] + hids + [l_coord, l_brush, l_out]
    return layers_from_list_to_dict(all_layers)


def model79(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    vertebrate with parallel convs
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * 3
    sparse_layers = []

    def sparse(l):
        name = l.name
        l = layers.NonlinearityLayer(
            l, wta_spatial,
            name="wta_spatial_{}".format(name))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
            l, wta_channel_strided(stride=4),
            name="wta_channel_{}".format(name))
        sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv1 = layers.Conv2DLayer(
        l_in,
        num_filters=nb_filters[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv1")
    l_conv1_sparse = sparse(l_conv1)
    l_conv2 = layers.Conv2DLayer(
        l_conv1,
        num_filters=nb_filters[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv2")
    l_conv2_sparse = sparse(l_conv2)
    l_conv3 = layers.Conv2DLayer(
        l_conv2,
        num_filters=nb_filters[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv3")
    l_conv3_sparse = sparse(l_conv3)

    l_conv3_back = l_conv3_sparse
    for i in range(2):
        shape = (l_conv3_back.output_shape[1], nb_filters[2 - i - 1], 5, 5)
        W = init.GlorotUniform().sample(shape)
        mask = np.zeros_like(W).astype(np.float32)
        k = np.arange(W.shape[0])
        mask[k, k, :, :] = 1
        W = theano.shared(W)
        W = W * mask
        l_conv3_back = layers.Conv2DLayer(
            l_conv3_back,
            num_filters=nb_filters[2 - i - 1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=W,
            pad='full'
        )

    l_conv2_back = l_conv2_sparse
    for i in range(1):
        shape = (l_conv2_back.output_shape[1], nb_filters[0], 5, 5)
        W = init.GlorotUniform().sample(shape)
        mask = np.zeros_like(W).astype(np.float32)
        k = np.arange(W.shape[0])
        mask[k, k, :, :] = 1
        W = theano.shared(W)
        W = W * mask
        l_conv2_back = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=W,
            pad='full'
        )
    l_conv1_back = l_conv1_sparse
    l_out1 = layers.Conv2DLayer(
        l_conv1_back,
        num_filters=c,
        filter_size=(5, 5),
        nonlinearity=linear,
        W=init.GlorotUniform(),
        pad='full',
        name='out1')
    l_out2 = layers.Conv2DLayer(
        l_conv2_back,
        num_filters=c,
        filter_size=(5, 5),
        nonlinearity=linear,
        W=init.GlorotUniform(),
        pad='full',
        name='out2')
    l_out3 = layers.Conv2DLayer(
        l_conv3_back,
        num_filters=c,
        filter_size=(5, 5),
        nonlinearity=linear,
        W=init.GlorotUniform(),
        pad='full',
        name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, T.add)
    l_out = layers.NonlinearityLayer(l_out, sigmoid, name='output')
    all_layers = ([l_in, l_conv1, l_conv2, l_conv3] +
                  sparse_layers +
                  [l_out1, l_out2, l_out3, l_out])
    return layers_from_list_to_dict(all_layers)

def model80(nb_filters=64, w=32, h=32, c=1, sparsity=True):
    """
    model67 but with occlusion
    """
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * 3
    sparse_layers = []

    def sparse(l):
        name = l.name
        l = layers.NonlinearityLayer(
                l, wta_spatial,
                name="wta_spatial_{}".format(name))
        sparse_layers.append(l)
        l = layers.NonlinearityLayer(
                l, wta_channel_strided(stride=4),
                name="wta_channel_{}".format(name))
        sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv1 = layers.Conv2DLayer(
            l_in,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv1")
    l_conv1_sparse = sparse(l_conv1)
    l_conv2 = layers.Conv2DLayer(
            l_conv1,
            num_filters=nb_filters[1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv2")
    l_conv2_sparse = sparse(l_conv2)
    l_conv3 = layers.Conv2DLayer(
             l_conv2,
             num_filters=nb_filters[2],
             filter_size=(5, 5),
             nonlinearity=rectify,
             W=init.GlorotUniform(),
             name="conv3")
    l_conv3_sparse = sparse(l_conv3)

    l_conv3_back = l_conv3_sparse
    for i in range(2):
        l_conv3_back = layers.Conv2DLayer(
            l_conv3_back,
            num_filters=nb_filters[2 - i - 1],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full'
        )

    l_conv2_back = l_conv2_sparse
    for i in range(1):
        l_conv2_back = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=nb_filters[0],
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            pad='full'
        )
    l_conv1_back = l_conv1_sparse
    l_out1 = layers.Conv2DLayer(
            l_conv1_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    l_out2 = layers.Conv2DLayer(
            l_conv2_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out2')
    l_out3 = layers.Conv2DLayer(
            l_conv3_back,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out3')
    out_layers = [l_out1, l_out2, l_out3]
    l_out = layers.ElemwiseMergeLayer(out_layers, over_op)
    l_out = layers.NonlinearityLayer(l_out, linear, name='output')
    all_layers = [l_in, l_conv1, l_conv2, l_conv3] + sparse_layers + [l_out1, l_out2, l_out3, l_out]
    return layers_from_list_to_dict(all_layers)


def model81(w=32, h=32, c=1,
            nb_fc_layers=3,
            nb_recurrent_layers=1,
            nb_recurrent_units=100,
            nb_fc_units=1000,
            nb_conv_layers=0,
            nb_conv_filters=64,
            size_conv_filters=3,
            n_steps=10,
            patch_size=3,
            w_out=-1,  # for up-scaling
            h_out=-1, # for up-scaling
            stride=True,  # if True, use strides else set them to 1
            sigma=None,  # if None, use sigma else set it to the given value  of sigma
            normalize='maxmin',  # ways to normalize : maxmin (like batch normalization but normalizes to 0..1)/sigmoid (applies sigmoid)/none
            pooling=True,
            alpha=0.5,
            reduce='sum', # ways to aggregate the brush layers : sum/over
            nonlin='rectify',
            theta=0.5,
            nonlin_brush='linear',
            coords_linear_layer=False,
            nonlin_out='sigmoid'):
    """

    model78 but with brush layer with
    return_seq = True and up-scaling and possibility to not have stride
    """

    def init_method():
        return init.GlorotUniform(gain='relu')
    if type(nb_fc_units) != list:
        nb_fc_units = [nb_fc_units] * nb_fc_layers

    if type(nb_conv_filters) != list:
        nb_conv_filters = [nb_conv_filters] * nb_conv_layers
    if type(size_conv_filters) != list:
        size_conv_filters = [size_conv_filters] * nb_conv_layers

    if type(nb_recurrent_units) != list:
        nb_recurrent_units = [nb_recurrent_units] * nb_recurrent_layers
    if w_out == -1:
        w_out = w
    if h_out == -1:
        h_out = h
    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = l_in
    nonlin = get_nonlinearity[nonlin]

    hids = []
    for i in range(nb_conv_layers):
        l_hid = layers.Conv2DLayer(
            l_hid,
            num_filters=nb_conv_filters[i],
            filter_size=(size_conv_filters[i], size_conv_filters[i]),
            nonlinearity=nonlin,
            W=init_method(),
            name='conv_hid{}'.format(i + 1))

        hids.append(l_hid)

        if pooling:
            l_hid = layers.Pool2DLayer(l_hid, (2, 2))

    for i in range(nb_fc_layers):
        l_hid = layers.DenseLayer(
            l_hid, nb_fc_units[i],
            W=init_method(),
            nonlinearity=nonlin,
            name="hid{}".format(i + 1))
        hids.append(l_hid)
    l_hid = Repeat(l_hid, n_steps)
    for i in range(nb_recurrent_layers):
        l_hid = layers.GRULayer(l_hid, nb_recurrent_units[i])

    if coords_linear_layer:
        l_coord = TensorDenseLayer(l_hid, 5, nonlinearity=linear, name="coord")
    else:
        l_coord = layers.GRULayer(l_hid, 5, name="coord")
    l_hid = layers.ReshapeLayer(l_coord, ([0], n_steps, 5), name="hid3")

    normalize_func = {'maxmin': norm_maxmin,
                      'sigmoid': T.nnet.sigmoid,
                      'none': lambda x: x}[normalize]
    reduce_func = {'sum': sum_op,
                   'over': over_op,
                   'normalized_over': normalized_over_op,
                   'max': max_op,
                   'thresh': thresh_op(theta),
                   'correct_over': correct_over_op(alpha)}[reduce]

    nonlin_brush = get_nonlinearity[nonlin_brush]
    l_brush = BrushLayer(
        l_hid,
        w_out, h_out,
        n_steps=n_steps,
        patch=np.ones((patch_size * (w_out/w), patch_size * (h_out/h) )),
        return_seq=True,
        stride=stride,
        sigma=sigma,
        normalize_func=normalize_func,
        reduce_func=reduce_func,
        nonlin_func=nonlin_brush,
        name="brush")
    l_out = layers.ExpressionLayer(l_brush, lambda x: x[:, -1, :, :], name="output", output_shape='auto')
    l_out = layers.ReshapeLayer(l_out, ([0], c, w_out, h_out), name="output")
    l_out_bias = layers.BiasLayer(
        l_out,
        b=init.Constant(-1.),
        name='bias') # because we are assuming the prev layer is between 0 and 1, we 'center' it at the beginning
    l_out = layers.NonlinearityLayer(
        l_out_bias,
        nonlinearity=get_nonlinearity[nonlin_out],
        name="output")
    all_layers = [l_in] + hids + [l_coord, l_brush, l_out_bias, l_out]
    return layers_from_list_to_dict(all_layers)


def model82(w=32, h=32, c=1,
            nb_recurrent_units=100,
            n_steps=10,
            patch_size=3,
            w_out=-1,
            h_out=-1,
            stride=True,
            sigma=None,
            normalize='maxmin',
            alpha=0.5,
            theta=0.5,

            out_reduce='sum',
            inp_reduce='sum',
            nonlin='rectify',
            nonlin_out='sigmoid',
            nb_fc_units=1000,
            normalize_patch=False,
            nb_fc_layers=0):

    """

    model81 but with input feedback
    """

    if type(nb_recurrent_units) != list:
        nb_recurrent_units = [nb_recurrent_units]

    if type(nb_fc_units) != list:
        nb_fc_units = [nb_fc_units] * nb_fc_layers

    def init_method():
        return init.GlorotUniform(gain='relu')
    if w_out == -1:
        w_out = w
    if h_out == -1:
        h_out = h

    normalize_func = {'maxmin': norm_maxmin,
                      'sigmoid': T.nnet.sigmoid,
                      'none': lambda x: x}[normalize]
    reduce_func = {'sum': sum_op,
                   'over': over_op,
                   'normalized_over': normalized_over_op,
                   'max': max_op,
                   'thresh': thresh_op(theta),
                   'prev': lambda a, b: a,
                   'new': lambda a, b: b,
                   'sub': lambda a, b: a - T.nnet.sigmoid(b),
                   'correct_over': correct_over_op(alpha)}

    l_in = layers.InputLayer((None, c, w, h), name="input")
    nonlin = get_nonlinearity[nonlin]

    patch = np.ones((patch_size * (w_out/w), patch_size * (h_out/h)))
    if normalize_patch:
        patch /= np.prod(patch.shape)

    brush_in = layers.InputLayer((None, nb_recurrent_units[0]))
    brush_in = layers.DenseLayer(brush_in, 5, nonlinearity=linear)
    brush_in = layers.ReshapeLayer(brush_in, ([0], 1, 5))
    l_brush = BrushLayer(
        brush_in,
        w_out, h_out,
        n_steps=1,
        patch=patch,
        return_seq=False,
        stride=stride,
        sigma=sigma,
        normalize_func=normalize_func,
        reduce_func=lambda prev, new: prev,
        nonlin_func=get_nonlinearity['linear'])
    l_brush_ = layers.ReshapeLayer(l_brush, ([0], w_out * h_out))

    def update_in(prev_inp, prev_out):
        return prev_inp

    def decorate_in(inp, prev_out):
        return reduce_func[inp_reduce](inp, downscale(prev_out))

    def downscale(x):
        if w_out > w and h_out > h:
            x = x.reshape((x.shape[0], w_out, h_out))
            x = x.reshape((x.shape[0], w_out / w, w, h_out / h, h))
            x = x.mean(axis=(1, 3))
            x = x.reshape((x.shape[0], w * h))
            return x
        else:
            return x

    def update_out(prev_out, new_out):
        return reduce_func[out_reduce](prev_out, new_out)

    l_in_ = layers.ReshapeLayer(l_in, ([0], w * h))

    in_repr = l_in
    hids = []
    for i in range(nb_fc_layers):
        in_repr = layers.DenseLayer(
            in_repr, nb_fc_units[i],
            W=init_method(),
            nonlinearity=nonlin,
            name="hid{}".format(i + 1))
        hids.append(in_repr)

    l_canvas = FeedbackGRULayer(
        l_in_,
        num_units=nb_recurrent_units[0],
        update_in=update_in,
        update_out=update_out,
        decorate_in=decorate_in,
        hid_to_out=l_brush_,
        in_to_repr=in_repr,
        n_steps=n_steps)
    l_canvas = layers.ReshapeLayer(
        l_canvas, ([0], n_steps, w_out, h_out), name="brush")
    l_out = layers.ExpressionLayer(
        l_canvas,
        lambda x: x[:, -1, :, :],
        name="output",
        output_shape='auto')

    l_raw_out = layers.ReshapeLayer(
        l_out,
        ([0], c, w_out, h_out),
        name="raw_output")
    l_scaled_out = layers.ScaleLayer(l_raw_out, scales=init.Constant(2.), name="scaled_output")
    l_biased_out = layers.BiasLayer(l_scaled_out, b=init.Constant(-1), name="biased_output")

    l_out = layers.NonlinearityLayer(
        l_biased_out,
        nonlinearity=get_nonlinearity[nonlin_out],
        name="output")
    all_layers = [l_in] + hids + [l_canvas, l_raw_out, l_scaled_out, l_biased_out, l_out]
    return layers_from_list_to_dict(all_layers)

def model83(w=32, h=32, c=1,
            nb_fc_layers=3,
            nb_recurrent_layers=1,
            nb_recurrent_units=100,
            nb_fc_units=1000,
            nb_conv_layers=0,
            nb_conv_filters=64,
            size_conv_filters=3,
            nonlin='relu',
            nonlin_out='sigmoid',
            pooling=True,
            n_steps=10,
            patch_size=3,
            w_out=-1,
            h_out=-1,
            reduce_func='sum',
            proba_func='sparsemax',
            normalize_func='sigmoid',
            x_sigma=0.5,
            y_sigma=0.5,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            recurrent_model='gru',
            variational=False,
            variational_nb_hidden=100,
            variational_seed=1,
            patches=None,
            col=None,
            eps=0):

    """
    the new GenericBrushLayer
    """
    ##  ENCODING part
    def init_method():
        return init.GlorotUniform(gain='relu')
    if type(nb_fc_units) != list:
        nb_fc_units = [nb_fc_units] * nb_fc_layers

    if type(nb_conv_filters) != list:
        nb_conv_filters = [nb_conv_filters] * nb_conv_layers
    if type(size_conv_filters) != list:
        size_conv_filters = [size_conv_filters] * nb_conv_layers

    if type(nb_recurrent_units) != list:
        nb_recurrent_units = [nb_recurrent_units] * nb_recurrent_layers
    if w_out == -1:
        w_out = w
    if h_out == -1:
        h_out = h

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_hid = l_in
    nonlin = get_nonlinearity[nonlin]

    hids = []
    for i in range(nb_conv_layers):
        l_hid = layers.Conv2DLayer(
            l_hid,
            num_filters=nb_conv_filters[i],
            filter_size=(size_conv_filters[i], size_conv_filters[i]),
            nonlinearity=nonlin,
            W=init_method(),
            name='conv_hid{}'.format(i + 1))
        hids.append(l_hid)

        if pooling:
            l_hid = layers.Pool2DLayer(l_hid, (2, 2), name='pool_hid{}'.format(i + 1))
            hids.append(l_hid)

    for i in range(nb_fc_layers):
        l_hid = layers.DenseLayer(
            l_hid, nb_fc_units[i],
            W=init_method(),
            nonlinearity=nonlin,
            name="hid{}".format(i + 1))
        hids.append(l_hid)

    if variational:
        z_mu = layers.DenseLayer(l_hid, variational_nb_hidden, nonlinearity=linear, name='z_mu')
        hids.append(z_mu)
        z_log_sigma = layers.DenseLayer(l_hid, variational_nb_hidden, nonlinearity=linear, name='z_log_sigma')
        hids.append(z_log_sigma)
        z = GaussianSampleLayer(
            z_mu, z_log_sigma,
            rng=RandomStreams(variational_seed),
            name='z_sample')
        hids.append(z)
        l_hid = z
    l_hid = Repeat(l_hid, n_steps)

    recurrent_model = recurrent_models[recurrent_model]
    for i in range(nb_recurrent_layers):
        l_hid = recurrent_model(l_hid, nb_recurrent_units[i])

    nb = (2 +  # coords +
          (1 if x_sigma == 'predicted' else 0) +
          (1 if y_sigma == 'predicted' else 0) +
          (1 if x_stride == 'predicted' else 0) +
          (1 if y_stride == 'predicted' else 0) +
          (c if color == 'predicted' else 0) +
          (1 if patch_index == 'predicted' else 0))
    nb = max(nb, 5)
    l_coord = TensorDenseLayer(l_hid, nb, nonlinearity=linear, name="coord")

    # DECODING PART
    
    if patches is None:
        patches = np.ones((1, c, patch_size * (h_out/h), patch_size * (w_out/w)))
        patches = patches.astype(np.float32)

    l_brush = GenericBrushLayer(
        l_coord,
        w_out, h_out,
        n_steps=n_steps,
        patches=patches,
        col=col if col else ('rgb' if c == 3 else 'grayscale'),
        return_seq=True,
        reduce_func=reduce_funcs[reduce_func],
        to_proba_func=proba_funcs[proba_func],
        normalize_func=normalize_funcs[normalize_func],
        x_sigma=x_sigma,
        y_sigma=y_sigma,
        x_stride=x_stride,
        y_stride=y_stride,
        patch_index=patch_index,
        color=color,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        eps=eps,
        name="brush"
    )
    l_raw_out = layers.ExpressionLayer(
        l_brush,
        lambda x: x[:, -1, :, :],
        name="raw_output",
        output_shape='auto')

    l_scaled_out = layers.ScaleLayer(
        l_raw_out, scales=init.Constant(2.), name="scaled_output")
    l_biased_out = layers.BiasLayer(
        l_scaled_out, b=init.Constant(-1), name="biased_output")

    l_out = layers.NonlinearityLayer(
        l_biased_out,
        nonlinearity=get_nonlinearity[nonlin_out],
        name="output")
    all_layers = ([l_in] +
                  hids +
                  [l_coord, l_brush, l_raw_out, l_biased_out, l_scaled_out,  l_out])
    return layers_from_list_to_dict(all_layers)

def model84( w=32, h=32, c=1, seed=42):
    """
    simple vae model with a fully connected neural net
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hid = layers.DenseLayer(l_in, 256, nonlinearity=rectify, name="hid")
    z_mu = layers.DenseLayer(hid, 2, nonlinearity=linear, name='z_mu')
    z_log_sigma = layers.DenseLayer(hid, 2, nonlinearity=linear, name='z_log_sigma')
    z = GaussianSampleLayer(z_mu, z_log_sigma, rng=RandomStreams(seed), name='z_sample')
    z_sample = z
    hid = layers.DenseLayer(z, 256, nonlinearity=rectify, name="hid")
    l_out = layers.DenseLayer(hid, c*w*h, nonlinearity=sigmoid, name='hid')
    l_out = layers.ReshapeLayer(l_out, ([0], c, w, h), name="output")
    return layers_from_list_to_dict([l_in, z_mu, z_log_sigma, z_sample, l_out])


def model85(w=32, h=32, c=1, n_steps=10, patch_size=5):
    """
    brush stroke where  I am experimenting with the halting unit
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hid = layers.DenseLayer(l_in, 800, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 700, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 800, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 300, nonlinearity=rectify, name="hid")


    hid = Repeat(hid, n_steps)
    hid = layers.LSTMLayer(hid, 400)
    l_coord = TensorDenseLayer(hid, 5, name="coord")

    patches = np.ones((1, c, patch_size, patch_size))
    patches = patches.astype(np.float32)

    l_brush = GenericBrushLayer(
        l_coord,
        w, h,
        n_steps=n_steps,
        patches=patches,
        col='rgb' if c == 3 else 'grayscale',
        return_seq=True,
        reduce_func=reduce_funcs['sum'],
        to_proba_func=proba_funcs['softmax'],
        normalize_func=normalize_funcs['sigmoid'],
        x_sigma=1,
        y_sigma=1,
        x_stride=1,
        y_stride=1,
        patch_index=0,
        color=[1.],
        x_min=0,
        x_max='width',
        y_min=0,
        y_max='height',
        name="brush"
    )
    def fn(brush, coord):
        halt =  T.nnet.sigmoid(coord[:, :, 2])
        halt = T.extra_ops.cumsum(halt, axis=1)
        halt = halt.dimshuffle(0, 1, 'x', 'x', 'x')
        return (halt * brush)[:, -1]

    l_raw_out = ExpressionLayerMulti(
        (l_brush, l_coord),
        fn,
        name="raw_output",
        output_shape=(l_brush.output_shape[0],) + l_brush.output_shape[2:])
    print(l_raw_out.output_shape)
    l_scaled_out = layers.ScaleLayer(
        l_raw_out, scales=init.Constant(2.), name="scaled_output")
    l_biased_out = layers.BiasLayer(
        l_scaled_out, b=init.Constant(-1), name="biased_output")

    l_out = layers.NonlinearityLayer(
        l_biased_out,
        nonlinearity=get_nonlinearity['sigmoid'],
        name="output")
    all_layers = ([l_in] +
                  [l_coord, l_brush, l_raw_out, l_biased_out, l_scaled_out,  l_out])
    return layers_from_list_to_dict(all_layers)



def model86(w=32, h=32, c=1, n_steps=10, patch_size=5):
    """
    brushstroke where i am experimenting with alpha blending prediction"
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hid = layers.DenseLayer(l_in, 800, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 700, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 800, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 300, nonlinearity=rectify, name="hid")

    hid = Repeat(hid, n_steps)
    hid = layers.LSTMLayer(hid, 400)
    l_coord = TensorDenseLayer(hid, 5, name="coord")

    patches = np.ones((1, c, patch_size, patch_size))
    patches = patches.astype(np.float32)

    l_brush = GenericBrushLayer(
        l_coord,
        w, h,
        n_steps=n_steps,
        patches=patches,
        col='rgb' if c == 3 else 'grayscale',
        return_seq=True,
        reduce_func=reduce_funcs['new'],
        to_proba_func=proba_funcs['softmax'],
        normalize_func=normalize_funcs['sigmoid'],
        x_sigma=1,
        y_sigma=1,
        x_stride=1,
        y_stride=1,
        patch_index=0,
        color=[1.],
        x_min=0,
        x_max='width',
        y_min=0,
        y_max='height',
        name="brush"
    )

    def fn(brush, coord):
        alpha = T.nnet.sigmoid(coord[:, :, 2])
        shape = brush.shape
        output_shape = (shape[0],) + (c, h, w)
        init_val = T.zeros(output_shape)
        init_val = T.unbroadcast(init_val, 0, 1, 2, 3)

        def step_function(input_cur, alpha_cur, output_prev):
            alpha_cur = alpha_cur.dimshuffle(0, 'x', 'x', 'x')
            return (output_prev * (1 - alpha_cur) + input_cur) # / (2 - alpha_cur)

        X = brush.transpose((1, 0, 2, 3, 4))
        alpha = alpha.transpose((1, 0))
        sequences = [X, alpha]
        outputs_info = [init_val]
        non_sequences = []
        result, updates = theano.scan(
            fn=step_function,
            sequences=sequences,
            outputs_info=outputs_info,
            non_sequences=non_sequences,
            strict=False,
            n_steps=n_steps)
        result = result[-1]
        return result

    l_raw_out = ExpressionLayerMulti(
        (l_brush, l_coord),
        fn,
        name="raw_output",
        output_shape=(l_brush.output_shape[0],) + l_brush.output_shape[2:])
    l_scaled_out = layers.ScaleLayer(
        l_raw_out, scales=init.Constant(2.), name="scaled_output")
    l_biased_out = layers.BiasLayer(
        l_scaled_out, b=init.Constant(-1), name="biased_output")

    l_out = layers.NonlinearityLayer(
        l_biased_out,
        nonlinearity=get_nonlinearity['sigmoid'],
        name="output")
    all_layers = ([l_in] +
                  [l_coord, l_brush, l_raw_out, l_biased_out, l_scaled_out,  l_out])
    return layers_from_list_to_dict(all_layers)

def model87(w=32, h=32, c=1, n_steps=10, patch_size=5):
    """
    brushstrike where I try to use sparse-max or softmax pixel wise to have a sharper result"
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hid = layers.DenseLayer(l_in, 800, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 700, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 800, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 300, nonlinearity=rectify, name="hid")

    hid = Repeat(hid, n_steps)
    hid = layers.LSTMLayer(hid, 400)
    l_coord = TensorDenseLayer(hid, 5, name="coord")

    patches = np.ones((1, c, patch_size, patch_size))
    patches = patches.astype(np.float32)

    l_brush = GenericBrushLayer(
        l_coord,
        w, h,
        n_steps=n_steps,
        patches=patches,
        col='rgb' if c == 3 else 'grayscale',
        return_seq=True,
        reduce_func=reduce_funcs['new'],
        to_proba_func=proba_funcs['softmax'],
        normalize_func=normalize_funcs['sigmoid'],
        x_sigma=1,
        y_sigma=1,
        x_stride=1,
        y_stride=1,
        patch_index=0,
        color=[1.],
        x_min=0,
        x_max='width',
        y_min=0,
        y_max='height',
        name="brush"
    )

    def fn(x):
        return softmax_(x, axis=1).sum(axis=1)

    l_raw_out = layers.ExpressionLayer(
        l_brush,
        fn,
        name="raw_output",
        output_shape="auto")

    l_scaled_out = layers.ScaleLayer(
        l_raw_out, scales=init.Constant(2.), name="scaled_output")
    l_biased_out = layers.BiasLayer(
        l_scaled_out, b=init.Constant(-1), name="biased_output")

    l_out = layers.NonlinearityLayer(
        l_biased_out,
        nonlinearity=get_nonlinearity['sigmoid'],
        name="output")
    all_layers = ([l_in] +
                  [l_coord, l_brush, l_raw_out, l_biased_out, l_scaled_out,  l_out])
    return layers_from_list_to_dict(all_layers)

def model88(w=32, h=32, c=1,
            nb_fc_layers=3,
            nb_recurrent_layers=1,
            nb_recurrent_units=100,
            nb_fc_units=1000,
            nb_conv_layers=0,
            nb_conv_filters=64,
            size_conv_filters=3,
            nonlin='relu',
            nonlin_out='sigmoid',
            pooling=True,
            n_steps=10,
            patch_size=3,
            w_out=-1,
            h_out=-1,
            reduce_func='sum',
            proba_func='sparsemax',
            normalize_func='sigmoid',
            x_sigma=0.5,
            y_sigma=0.5,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            nb_patches=1,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            recurrent_model='gru',
            learn_patches=False,
            variational=False,
            variational_nb_hidden=100,
            variational_seed=1,
            patches=None,
            col=None,
            parallel=1,
            parallel_share=True,
            parallel_reduce_func='sum',
            w_left_pad=0,
            w_right_pad=0,
            h_left_pad=0,
            h_right_pad=0,
            color_min=0,
            color_max=1,
            stride_normalize=False,
            coords='continuous',
            learn_bias_scale=True,
            eps=0):

    """
    a clean version of GenericBrushLayer
    """

    # INIT
    def init_method(): return init.GlorotUniform(gain='relu')
    if type(nb_fc_units) != list: nb_fc_units = [nb_fc_units] * nb_fc_layers
    if type(nb_conv_filters) != list: nb_conv_filters = [nb_conv_filters] * nb_conv_layers
    if type(size_conv_filters) != list: size_conv_filters = [size_conv_filters] * nb_conv_layers
    if type(nb_recurrent_units) != list: nb_recurrent_units = [nb_recurrent_units] * nb_recurrent_layers
    if w_out == -1: w_out = w
    if h_out == -1: h_out = h
    output_shape = (None, c, h_out, w_out)
    nonlin = get_nonlinearity[nonlin]
    recurrent_model = recurrent_models[recurrent_model]
    if patches is None:patches = np.ones((nb_patches, c, patch_size * (h_out/h), patch_size * (w_out/w)));patches = patches.astype(np.float32)
    extra_layers = []

    ##  ENCODING part
    in_ = layers.InputLayer((None, c, w, h), name="input")
    hid = in_
        
    nets = []
    for i in range(parallel):
        hids = conv_fc(
          hid, 
          num_filters=nb_conv_filters, 
          size_conv_filters=size_conv_filters, 
          init_method=init_method,
          pooling=pooling,
          nb_fc_units=nb_fc_units,
          nonlin=nonlin,
          names_prefix='net{}'.format(i))
        nets.append(hids)
   
    if variational:
        all_z_mu = []
        all_z_log_sigma = []
        for n, net in enumerate(nets):
            z_mu = layers.DenseLayer(net[-1], variational_nb_hidden, nonlinearity=linear, name='z_mu_{}'.format(n))
            z_log_sigma = layers.DenseLayer(net[-1], variational_nb_hidden, nonlinearity=linear, name='z_log_sigma_{}'.format(n))
            all_z_mu.append(z_mu)
            all_z_log_sigma.append(z_log_sigma)
            net.append(z_mu)
            net.append(z_log_sigma)

        z_mu = layers.ConcatLayer(all_z_mu, axis=1, name='z_mu')
        z_log_sigma = layers.ConcatLayer(all_z_log_sigma, axis=1, name='z_log_sigma')
        z = GaussianSampleLayer(
            z_mu, z_log_sigma,
            rng=RandomStreams(variational_seed),
            name='z_sample')

        extra_layers.append(z_mu)
        extra_layers.append(z_log_sigma)
        extra_layers.append(z)
        i = 0
        for n, net in enumerate(nets):
            z_net = layers.SliceLayer(z, indices=slice(i, i + variational_nb_hidden), axis=1, name='z_sample_{}'.format(n))
            i += variational_nb_hidden
            net.append(z_net)

    for n, net in enumerate(nets):
        hid = Repeat(net[-1], n_steps)
        for l in range(nb_recurrent_layers):
            hid = recurrent_model(
                    hid, 
                    nb_recurrent_units[l], 
                    name="recurrent{}_{}".format(l, n))
        net.append(hid)
    nb = ( (2 if coords == 'continuous' else (w+h) if coords == 'discrete' else 0) +
          (1 if x_sigma == 'predicted' else 0) +
          (len(x_sigma) if type(x_sigma) == list else 0) + 
          (1 if y_sigma == 'predicted' else 0) +
          (len(y_sigma) if type(y_sigma) == list else 0) + 
          (1 if x_stride == 'predicted' else 0) +
          (len(x_stride) if type(x_stride) == list else 0) + 
          (1 if y_stride == 'predicted' else 0) +
          (len(y_stride) if type(y_stride) == list else 0) + 
          (c if color == 'predicted' else 0) +
          (1 if patch_index == 'predicted' else 0) +
          (nb_patches if patch_index == 'predicted' else 0) +
          (color if type(color) == int else 0)
    )
    if type(color) == int:
        color = np.random.normal(0.01, size=(color, c)).astype(np.float32)
    for i, net in enumerate(nets):
        hid = net[-1]
        coord = TensorDenseLayer(hid, nb, nonlinearity=linear, name="coord_{}".format(i))
        net.append(coord)

    # DECODING PART
    
    for i, net in enumerate(nets):
        coord = net[-1]
        brush = GenericBrushLayer(
            coord,
            w_out, h_out,
            n_steps=n_steps,
            patches=patches,
            col=col if col else ('rgb' if c == 3 else 'grayscale'),
            return_seq=True,
            reduce_func=reduce_funcs[reduce_func],
            to_proba_func=proba_funcs[proba_func],
            normalize_func=normalize_funcs[normalize_func],
            x_sigma=x_sigma,
            y_sigma=y_sigma,
            x_stride=x_stride,
            y_stride=y_stride,
            patch_index=patch_index,
            learn_patches=learn_patches,
            color=color,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            w_left_pad=w_left_pad,
            w_right_pad=w_right_pad,
            h_left_pad=h_left_pad,
            h_right_pad=h_right_pad,
            color_min=color_min,
            color_max=color_max,
            stride_normalize=stride_normalize,
            coords=coords,
            eps=eps,
            name="brush_{}".format(i)
        )
        net.append(brush)

    for i, net in enumerate(nets):
        brush = net[-1]
        raw_out = layers.ExpressionLayer(
            brush,
            lambda x: x[:, -1, :, :],
            name="raw_output_{}".format(i),
            output_shape='auto')
        net.append(brush)
    
    def nets_reduce(*nets):
        nets = map(lambda k:k[:, -1], nets) # get last time step output from each net
        func = reduce_funcs[parallel_reduce_func]
        return reduce(func, nets)

    raw_out = ExpressionLayerMulti(
        map(lambda net:net[-1], nets),
        nets_reduce,
        name="raw_output",
        output_shape=output_shape)
    
    if learn_bias_scale:
        scaled_out = layers.ScaleLayer(
            raw_out, scales=init.Constant(2.), name="scaled_output")
        biased_out = layers.BiasLayer(
            scaled_out, b=init.Constant(-1), name="biased_output")
    else:
        scaled_out = raw_out
        biased_out = raw_out
    out = layers.NonlinearityLayer(
        biased_out,
        nonlinearity=get_nonlinearity[nonlin_out],
        name="output")
    all_layers = ([in_] +
                  [lay for net in nets for lay in net] +
                  extra_layers + 
                  [raw_out, scaled_out, biased_out, out])
    return layers_from_list_to_dict(all_layers)

def conv_fc(x, 
            num_filters=[32, 32], 
            size_conv_filters=[5, 5], 
            init_method=init.GlorotUniform,
            pooling=False,
            nb_fc_units=[100],
            nonlin=get_nonlinearity['rectify'],
            names_prefix=''):
    l_hid = x
    hids = []
    for i in range(len(num_filters)):
        l_hid = layers.Conv2DLayer(
            l_hid,
            num_filters=num_filters[i],
            filter_size=(size_conv_filters[i], size_conv_filters[i]),
            nonlinearity=nonlin,
            W=init_method(),
            name='conv{}_{}'.format(i + 1, names_prefix))
        hids.append(l_hid)
        if pooling:
            l_hid = layers.Pool2DLayer(l_hid, (2, 2), name='pool{}_{}'.format(i + 1, names_prefix))
            hids.append(l_hid)

    for i in range(len(nb_fc_units)):
        l_hid = layers.DenseLayer(
            l_hid, nb_fc_units[i],
            W=init_method(),
            nonlinearity=nonlin,
            name="fc{}_{}".format(i + 1, names_prefix))
        hids.append(l_hid)
    return hids

def model89(w=32, h=32, c=1, scale_min=8, n_steps_min=4, patch_size_min=1):
    """
    big brush stroke model where we divide the image into a grid of 4
    then divide each subgrid into a grid 4 and have a brush stroke model
    in each cell
    """
    from itertools import chain
    w_in = w
    h_in = h
    w_out = h
    w_out = w
    h_out = h
    nb_conv_filters = []
    size_conv_filters = []
    init_method = init.GlorotUniform
    pooling = False
    nb_fc_units = [500]
    nonlin = 'relu'

    in_ = layers.InputLayer((None, c, w_in, h_in), name="input")
    nets = []
    recurrent_model = layers.GRULayer
    brushes = {}
    for y, x, h, w in chain(iterate_scales(w=w_out, h=h_out), [(0, 0, h_out, w_out)]):
        if w < scale_min: continue
        print(y, x, h, w)
        scale = w_out / scale_min
        n_steps = n_steps_min * scale
        patch_h = patch_size_min * scale
        patch_w = patch_size_min * scale
        patches = np.ones((1, c, patch_h, patch_w), dtype='float32')

        hid = in_
        hid = layers.SliceLayer(hid, slice(y, y+h), axis=2)
        hid = layers.SliceLayer(hid, slice(x, x+w), axis=3)
        hid = layers.DenseLayer(hid, 100*scale, nonlinearity=rectify)
        hid = Repeat(hid, n_steps) 
        hid = recurrent_model(hid, 100)
        coord = TensorDenseLayer(hid, 5, nonlinearity=linear)
        brush = GenericBrushLayer(
            coord,
            w, h,
            n_steps=n_steps,
            patches=patches,
            col=('rgb' if c == 3 else 'grayscale'),
            return_seq=False,
            reduce_func=over_op,
            to_proba_func=softmax_,
            normalize_func=T.nnet.sigmoid,
            x_sigma=1,
            y_sigma=1,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            eps=0
        )
        brushes[(y, x, h, w)] = brush
    op = max_op
    def get_val(v=None, x=0, y=0, w=32, h=32):
        if w<=scale_min:
            return v
        else:
            t = [
                get_val(brushes[(y, x, h/2, w/2)], x=x, y=y, h=h/2, w=w/2),
                get_val(brushes[(y, x + w/2, h/2, w/2)], x=x+w/2, y=y, h=h/2, w=w/2),
                get_val(brushes[(y+h/2, x, h/2, w/2)], x=x,y=y+h/2, h=h/2, w=w/2),
                get_val(brushes[(y+h/2, x+w/2, h/2, w/2)], x=x+w/2,y=y+h/2, h=h/2,w=w/2)]
            if v:
                v = ExpressionLayerMulti([v, merge_scale(t)], op)
                return v
            else:
                return merge_scale(t)

    raw_out = get_val(v=brushes[(0, 0, h, w)], x=0, y=0, w=w_out, h=h_out)
    raw_out.name = "raw_output"
    scaled_out = layers.ScaleLayer(
        raw_out, scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(
        scaled_out, b=init.Constant(-1), name="biased_output")
    out = layers.NonlinearityLayer(
        biased_out,
        nonlinearity=T.nnet.sigmoid,
        name="output")
    all_layers = ([in_] +
                  [raw_out, scaled_out, biased_out, out])
    return layers_from_list_to_dict(all_layers)


def model89(w=32, h=32, c=1, scale_min=8, n_steps_min=4, patch_size_min=1):
    """
    brush stroke with iterative rescaling
    """
    nb_conv_filters = []
    size_conv_filters = []
    init_method = init.GlorotUniform
    pooling = False
    nb_fc_units = [500]
    nonlin = 'relu'
    nb_scales = int(np.log2(w / scale_min))
    recurrent_model = layers.RecurrentLayer
    
    in_ = layers.InputLayer((None, c, w, h), name="input")
    in_flat = layers.ReshapeLayer(in_, ([0], c*w*h))
    
    brushes = []
    coords = []
    scales_out = []
    w_cur, h_cur = scale_min, scale_min
    for i in range(nb_scales + 1):
        n_steps = n_steps_min
        if i > 0:
            x = layers.ConcatLayer((in_flat, brush_flat), axis=1)
        else:
            x = in_flat
        hid = layers.DenseLayer(x, 1000, nonlinearity=very_leaky_rectify)
        hid = layers.DenseLayer(hid, 1000, nonlinearity=very_leaky_rectify)
        
        hid = Repeat(hid, n_steps) 
        hid = recurrent_model(hid, 256)
        coord = TensorDenseLayer(hid, 7, nonlinearity=linear)
        coords.append(coord)

        patch_h = patch_size_min
        patch_w = patch_size_min
        patches = np.ones((1, c, patch_h, patch_w), dtype='float32')

        brush = GenericBrushLayer(
            coord,
            w_cur, h_cur,
            n_steps=n_steps,
            patches=patches,
            col=('rgb' if c == 3 else 'grayscale'),
            return_seq=False,
            reduce_func=sum_op,
            to_proba_func=softmax_,
            normalize_func=T.nnet.sigmoid,
            x_sigma=1,
            y_sigma=1,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            eps=0
        )
        brushes.append(brush)
        brush_flat = layers.ReshapeLayer(brush, ([0], c * w_cur * h_cur))
        if w == w_cur:
            scale_out = brush
        else:
            scale_out = layers.Upscale2DLayer(brush, w / w_cur)
        print(scale_out.output_shape)
        scales_out.append(scale_out)
        w_cur *= 2
        h_cur *= 2
    
    raw_out = scales_out[0]
    op = sum_op
    for o in scales_out[1:]:
        raw_out = ExpressionLayerMulti((raw_out, o), op)

    raw_out.name = "raw_output"
    scaled_out = layers.ScaleLayer(
        raw_out, scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(
        scaled_out, b=init.Constant(-1), name="biased_output")
    out = layers.NonlinearityLayer(
        biased_out,
        nonlinearity=T.nnet.sigmoid,
        name="output")
    all_layers = ([in_] +
                  #coords + 
                  #brushes + 
                  [raw_out, scaled_out, biased_out, out])
    return layers_from_list_to_dict(all_layers)

def model90(w=32,h=32,c=1, nb_filters=None, sparsity_second=True):
    """
    vertebrate model to force the neural net to learn compositionality
    because the number of low level filters are small.
    """
    if nb_filters is None: nb_filters=[64, 32, 8]
    nbf = nb_filters
    in_ = layers.InputLayer((None, c, w, h), name="input")
    conv1 = layers.Conv2DLayer(
        in_,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv1")
    conv2 = layers.Conv2DLayer(
        conv1,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv2")
    conv3 = layers.Conv2DLayer(
        conv2,
        num_filters=nbf[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv3")

    wta1 = layers.NonlinearityLayer(conv3, wta_spatial, name="wta1")
    wta2 = layers.NonlinearityLayer(wta1, linear, name="wta2")

    conv4 = layers.Conv2DLayer(
        conv3,
        num_filters=nbf[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv4")

    conv5 = layers.Conv2DLayer(
        conv4,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv5")

    conv6 = layers.Conv2DLayer(
        conv5,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv6")

    if sparsity_second: wta3 = layers.NonlinearityLayer(conv6, wta_spatial, name="wta3")
    else: wta3 = layers.NonlinearityLayer(conv6, linear, name="wta3")
    
    wta4 = layers.NonlinearityLayer(wta3, linear, name="wta4")

    conv7 = layers.Conv2DLayer(
        wta4,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv7")

    conv8 = layers.Conv2DLayer(
        conv7,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv8")

    conv9 = layers.Conv2DLayer(
        conv8,
        num_filters=nbf[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv9")
    out1  = layers.Conv2DLayer(
            wta2,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')
    out2  = layers.Conv2DLayer(
            conv9,
            num_filters=c,
            filter_size=(13, 13),
            nonlinearity=linear,
            W=out1.W,
            pad='full',
            name='out2')
    raw_out = layers.ElemwiseMergeLayer([out1, out2], T.add)
    scaled_out = layers.ScaleLayer(raw_out,  scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(scaled_out, b=init.Constant(-1),   name="biased_output")
    out = layers.NonlinearityLayer(biased_out, nonlinearity=sigmoid, name='output')
    convs = [
        conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9
    ]
    wtas = [wta1, wta2, wta3, wta4]
    return layers_from_list_to_dict([in_] + convs + wtas + [out1, out2, scaled_out, biased_out, out])


def model91(w=32,h=32,c=1, nb_filters=None):
    """
    model90 but allow weight sharing
    """
    if nb_filters is None: nb_filters=[16, 8, 16, 32]
    nbf = nb_filters
    in_ = layers.InputLayer((None, c, w, h), name="input")

    # DETECT IN FIRST SCALE
    conv1 = layers.Conv2DLayer(
        in_,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv1")
    conv2 = layers.Conv2DLayer(
        conv1,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv2")

    # SPARSE IN FIRST SCALE
    wta1 = layers.NonlinearityLayer(conv2, wta_spatial, name="wta1")
    wta2 = layers.NonlinearityLayer(wta1, linear, name="wta2")
        
    # DETECT IN SECOND SCALE
    conv3 = layers.Conv2DLayer(
        conv2,
        num_filters=nbf[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv3")

    conv4 = layers.Conv2DLayer(
        conv3,
        num_filters=nbf[3],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv4")

    # SPARSE IN SECOND SCALE
    wta3 = layers.NonlinearityLayer(conv4, wta_spatial, name="wta3")
    wta4 = layers.NonlinearityLayer(wta3, linear, name="wta4")
    
    # CONVERT SECOND TO FIRST  SCALE
    conv5 = layers.Conv2DLayer(
        wta4,
        num_filters=nbf[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv5")

    conv6 = layers.Conv2DLayer(
        conv5,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv6")

    conv7 = layers.Conv2DLayer(
        conv6,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv7")

    conv8 = layers.Conv2DLayer(
        conv7,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=conv2.W,
        name="conv8")

    out1  = layers.Conv2DLayer(
            wta2,
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')

    out2  = layers.Conv2DLayer(
            conv8,
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=linear,
            W=out1.W,
            pad='full',
            name='out2')

    raw_out = layers.ElemwiseMergeLayer([out1, out2], T.add)
    scaled_out = layers.ScaleLayer(raw_out,  scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(scaled_out, b=init.Constant(-1),   name="biased_output")
    out = layers.NonlinearityLayer(biased_out, nonlinearity=sigmoid, name='output')
    convs = [
        conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8
    ]
    wtas = [wta1, wta2, wta3, wta4]
    return layers_from_list_to_dict([in_] + convs + wtas + [out1, out2, scaled_out, biased_out, out])

def model92(w=32,h=32,c=1, nb_filters=None):
    """
    model91 but with more scales
    """
    if nb_filters is None: nb_filters=[16, 8, 32, 16, 64, 32]
    nbf = nb_filters
    in_ = layers.InputLayer((None, c, w, h), name="input")

    # DETECT IN FIRST SCALE
    conv1 = layers.Conv2DLayer(
        in_,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv1")

    conv2 = layers.Conv2DLayer(
        conv1,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv2")

    # SPARSE IN FIRST SCALE
    
    wta1 = layers.NonlinearityLayer(conv2, wta_spatial, name="wta1")

    # DETECT IN SECOND SCALE
    conv3 = layers.Conv2DLayer(
        conv2,
        num_filters=nbf[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv3")

    conv4 = layers.Conv2DLayer(
        conv3,
        num_filters=nbf[3],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv4")

    # SPARSE IN SECOND SCALE
    wta2 = layers.NonlinearityLayer(conv4, wta_spatial, name="wta2")

    # GO BACK TO FIRST SCALE

    conv5 = layers.Conv2DLayer(
        wta2,
        num_filters=nbf[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv5")

    conv6 = layers.Conv2DLayer(
        conv5,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv6")
    
    # DETECT IN THIRD SCALE

    conv7 = layers.Conv2DLayer(
        conv4,
        num_filters=nbf[4],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv7")

    conv8 = layers.Conv2DLayer(
        conv7,
        num_filters=nbf[5],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv8")

    # SPARSE IN THIRD SCALE
    wta3 = layers.NonlinearityLayer(conv8, wta_spatial, name="wta3")

    # GO BACK TO SECOND SCALE

    conv9 = layers.Conv2DLayer(
        wta3,
        num_filters=nbf[4],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv9")

    conv10 = layers.Conv2DLayer(
        conv9,
        num_filters=nbf[3],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv10")
 
    # GO BACK TO FIRST SCALE

    conv11 = layers.Conv2DLayer(
        conv10,
        num_filters=nbf[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=conv5.W,
        pad='full',
        name="conv11")

    conv12 = layers.Conv2DLayer(
        conv11,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=conv6.W,
        pad='full',
        name="conv12")
    print(wta1.output_shape, conv6.output_shape, conv12.output_shape)
    out1  = layers.Conv2DLayer(
            wta1,
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')

    out2  = layers.Conv2DLayer(
            conv6,
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=linear,
            W=out1.W,
            pad='full',
            name='out2')

    out3  = layers.Conv2DLayer(
            conv12,
            num_filters=c,
            filter_size=(9, 9),
            nonlinearity=linear,
            W=out1.W,
            pad='full',
            name='out3')
 
    raw_out = layers.ElemwiseMergeLayer([out1, out2, out3], T.add)
    scaled_out = layers.ScaleLayer(raw_out,  scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(scaled_out, b=init.Constant(-1),   name="biased_output")
    out = layers.NonlinearityLayer(biased_out, nonlinearity=sigmoid, name='output')
    convs = [
        conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
    ]
    wtas = [wta1, wta2, wta3]
    return layers_from_list_to_dict([in_] + convs + wtas + [out1, out2, out3, scaled_out, biased_out, out])

def model93(w=32, h=32,c=1, nb_filters=None):
    """
    vertebrate but one scale = one conv and a wta in the output layer
    for each scale
    """
    if nb_filters is None: nb_filters = [8, 32]
    nbf = nb_filters
    in_ = layers.InputLayer((None, c, w, h), name="input")

    conv1 = layers.Conv2DLayer(
        in_,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv1")

    wta1 = layers.NonlinearityLayer(conv1, wta_spatial, name="wta1")

    conv2 = layers.Conv2DLayer(
        conv1,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv2")

    wta2 = layers.NonlinearityLayer(conv2, wta_spatial, name="wta2")

    conv3 = layers.Conv2DLayer(
        wta2,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        pad='full',
        name="conv3")

    wta3 = layers.NonlinearityLayer(conv3, wta_k_spatial(4), name="wta3")
    print(wta1.output_shape, wta2.output_shape)
    out1  = layers.Conv2DLayer(
            wta1,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')

    out2  = layers.Conv2DLayer(
            wta3,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=out1.W,
            pad='full',
            name='out2')

    raw_out = layers.ElemwiseMergeLayer([out1, out2], T.add)
    scaled_out = layers.ScaleLayer(raw_out,  scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(scaled_out, b=init.Constant(-1),   name="biased_output")
    out = layers.NonlinearityLayer(biased_out, nonlinearity=sigmoid, name='output')
    convs = [
        conv1, conv2, conv3
    ]
    wtas = [wta1, wta2, wta3]
    return layers_from_list_to_dict([in_] + convs + wtas + [out1, out2, scaled_out, biased_out, out])


def model94(w=32, h=32,c=1, nb_filters=None):
    """
    model93 with one more scale
    """
    if nb_filters is None: nb_filters = [8, 32, 64]
    nbf = nb_filters
    in_ = layers.InputLayer((None, c, w, h), name="input")

    conv1 = layers.Conv2DLayer(
        in_,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv1")

    wta1 = layers.NonlinearityLayer(conv1, wta_spatial, name="wta1")

    conv2 = layers.Conv2DLayer(
        conv1,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv2")

    wta2 = layers.NonlinearityLayer(conv2, wta_spatial, name="wta2")

    conv3 = layers.Conv2DLayer(
        wta2,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        pad='full',
        name="conv3")

    wta3 = layers.NonlinearityLayer(conv3, wta_k_spatial(4), name="wta3")

    conv4 = layers.Conv2DLayer(
        conv2,
        num_filters=nbf[2],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name="conv4")

    wta4 = layers.NonlinearityLayer(conv3, wta_spatial, name="wta4")

    conv5 = layers.Conv2DLayer(
        conv4,
        num_filters=nbf[1],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv4")

    conv6 = layers.Conv2DLayer(
        conv5,
        num_filters=nbf[0],
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        pad='full',
        name="conv6")

    wta5 = layers.NonlinearityLayer(conv6, wta_k_spatial(8), name="wta5")

    out1  = layers.Conv2DLayer(
            wta1,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=init.GlorotUniform(),
            pad='full',
            name='out1')

    out2  = layers.Conv2DLayer(
            wta3,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=out1.W,
            pad='full',
            name='out2')

    out3  = layers.Conv2DLayer(
            wta5,
            num_filters=c,
            filter_size=(5, 5),
            nonlinearity=linear,
            W=out1.W,
            pad='full',
            name='out3')

    raw_out = layers.ElemwiseMergeLayer([out1, out2, out3], T.add)
    scaled_out = layers.ScaleLayer(raw_out,  scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(scaled_out, b=init.Constant(-1),   name="biased_output")
    out = layers.NonlinearityLayer(biased_out, nonlinearity=sigmoid, name='output')
    convs = [
        conv1, conv2, conv3, conv4, conv5, conv6
    ]
    wtas = [wta1, wta2, wta3, wta4, wta5]
    return layers_from_list_to_dict([in_] + convs + wtas + [out1, out2, scaled_out, biased_out, out])


def merge_scale(nets):
    # take 4 nets of shape (example, c, h, w) and returns their
    # concatenation in a grid of size (example, c, h * 2, h * 2)
    n1 = layers.ConcatLayer((nets[0], nets[1]), axis=3)
    n2 = layers.ConcatLayer((nets[2], nets[3]), axis=3)
    n = layers.ConcatLayer((n1, n2), axis=2)
    return n

def iterate_scales(x=0, y=0,w=32, h=32):
    S = 2
    if w <= 2 or h <= 2:
        return
    ts = [
        (y, x, h/S, w/S),
        (y, x+w/S, h/S, w/S),
        (y+h/S, x, h/S, w/S),
        (y+h/S, x+w/S, h/S, w/S)
    ]

    t = ts[0]
    for t_cur in iterate_scales(y=t[0], x=t[1], h=t[2], w=t[3]):
        yield t_cur
    t = ts[1]
    for t_cur in iterate_scales(y=t[0], x=t[1], h=t[2], w=t[3]):
        yield t_cur
    t = ts[2]
    for t_cur in iterate_scales(y=t[0], x=t[1], h=t[2], w=t[3]):
        yield t_cur
    t = ts[3]
    for t_cur in iterate_scales(y=t[0], x=t[1], h=t[2], w=t[3]):
        yield t_cur
    for t in ts:
        yield t


def model95(w=32, h=32,c=1):
    """
    """
    lays = []
    in_ = layers.InputLayer((None, c, w, h), name="input")

    conv = layers.Conv2DLayer(
        in_,
        num_filters=128,
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='conv1')
    lays.append(conv)

    conv = layers.Conv2DLayer(
        conv,
        num_filters=128,
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='conv2')
    lays.append(conv)
    conv = layers.Conv2DLayer(
        conv,
        num_filters=128,
        filter_size=(5, 5),
        nonlinearity=rectify,
        W=init.GlorotUniform(),
        name='conv3')
    lays.append(conv)

    nb_comp = [6, 3]
    dim_comp = [10, 10]
     
    Drepr = layers.DenseLayer(conv, nb_comp[0] * dim_comp[0], nonlinearity=linear, name='drepr')
    lays.append(Drepr)
    Drepr = layers.ReshapeLayer(Drepr, ([0], nb_comp[0], dim_comp[0]))
    Drepr = layers.ExpressionLayer(Drepr, lambda x:sparsemax_seq(x), output_shape='auto', name='drepr_normalized')
    lays.append(Drepr)

    Dcoord = layers.DenseLayer(conv, nb_comp[0] * 2, nonlinearity=linear, name='dcoord')
    lays.append(Dcoord)
    
    Dcoord = layers.ReshapeLayer(Dcoord, ([0], nb_comp[0], 2))
    Dcoord = layers.NonlinearityLayer(Dcoord, T.nnet.sigmoid, name='dcoord_normalized')
    lays.append(Dcoord)

    L = []
    W = init.GlorotUniform()
    b = init.Constant(0.)
    brushes = []

    patches = np.ones((1, 1, 3, 3)).astype(np.float32)
    for i in range(nb_comp[0]):
        dcoordi = layers.SliceLayer(Dcoord, i, axis=1)
        drepri = layers.SliceLayer(Drepr, i, axis=1)
        di = layers.ConcatLayer((dcoordi, drepri), axis=1)

        dcoordj = layers.DenseLayer(di, nb_comp[1] * 2, W=W, b=b, nonlinearity=linear, name='dcoordj_{}'.format(i))
        lays.append(dcoordj)
        W = dcoordj.W
        b = dcoordj.b
        
        dcoordj = layers.NonlinearityLayer(dcoordj, T.nnet.sigmoid)
        dcoordj = layers.ReshapeLayer(dcoordj, ([0], nb_comp[1], 2))

        dcoordj = ExpressionLayerMulti((dcoordi, dcoordj), lambda a, b:a[:, None, :] * b, output_shape=dcoordj.output_shape, name='dcoordj_normalized_{}'.format(i))
        lays.append(dcoordj)

        brush_i = GenericBrushLayer(
                dcoordj, w, h,
                patches=patches,
                learn_patches=True,
                col='grayscale',
                n_steps=nb_comp[1],
                return_seq=False,
                reduce_func=sum_op,
                to_proba_func=T.nnet.softmax,
                normalize_func=linear,
                x_sigma=0.5,
                y_sigma=0.5,
                x_stride=1,
                y_stride=1,
                patch_index=0,
                color=[1.],
                x_min=0,
                x_max='width',
                y_min=0,
                y_max='height',
                name='brush{}'.format(i),
                eps=0)
        patches = brush_i.patches_
        lays.append(brush_i)
        brushes.append(brush_i)
    raw_out = layers.ElemwiseSumLayer(brushes, name="raw_out")
    scaled_out = layers.ScaleLayer(
       raw_out, scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(
      scaled_out, b=init.Constant(-1), name="biased_output")
    out = layers.NonlinearityLayer(
       biased_out,
       nonlinearity=T.nnet.sigmoid,
       name="output")
    return layers_from_list_to_dict([in_] + lays + [raw_out, scaled_out, biased_out, out])

def model96(w=32, h=32,c=1, nb_comp=[3, 3, 3, 3], dim_comp=[10, 10, 10, 10], scales=[0.5, 0.2, 0.1, 0.01], nb_patches=1, patch_size=1):
    """
    """

    lays = []
    in_ = layers.InputLayer((None, c, w, h), name="input")
    hid = layers.DenseLayer(in_, 500, nonlinearity=rectify, name='hid')
    lays.append(hid)
    conv = hid

    brushes = []
    patches_all_brushes = np.ones((nb_patches, 1, patch_size, patch_size)).astype(np.float32)
    params_per_depth = {}
    
    #selector = lambda x:x
    #selector = sparsemax_seq
    selector = is_max_seq
    #selector = softmax_seq
    def add_program_layer(lrepr=None, lcoord=None, depth=0):
        nb_comp_cur = nb_comp[depth]
        nb_dim_cur = dim_comp[depth]
        if depth == 0:
            lrepr = layers.DenseLayer(conv, nb_comp[0] * dim_comp[0], nonlinearity=linear, name='repr_0')
            lays.append(lrepr)
            #lrepr = batch_norm(lrepr)

            lrepr = layers.ReshapeLayer(lrepr, ([0], nb_comp[0], dim_comp[0]))
            lrepr = layers.ExpressionLayer(lrepr, lambda x:selector(x), output_shape='auto', name='repr_0_normalized')
            lays.append(lrepr)

            lcoord = layers.DenseLayer(conv, nb_comp[0] * 2, nonlinearity=linear, name='coord_0')
            #lcoord = batch_norm(lcoord)
            lays.append(lcoord)
            
            lcoord = layers.ReshapeLayer(lcoord, ([0], nb_comp[0], 2))
            lcoord = layers.NonlinearityLayer(lcoord, T.nnet.sigmoid, name='coord_0_normalized')
            lays.append(lcoord)

        if depth == len(nb_comp) - 1:
            if len(brushes):
                patches = brushes[0].patches_
            else:
                patches = patches_all_brushes
            patches = patches_all_brushes
            brush = GenericBrushLayer(
                    lcoord, w, h,
                    patches=patches,
                    learn_patches=True,
                    col='grayscale',
                    n_steps=nb_comp_cur,
                    return_seq=False,
                    reduce_func=sum_op,
                    to_proba_func=T.nnet.softmax,
                    normalize_func=linear,
                    x_sigma=0.5,
                    y_sigma=0.5,
                    x_stride=1,
                    y_stride=1,
                    patch_index=0,
                    color=[1.],
                    x_min=0,
                    x_max='width',
                    y_min=0,
                    y_max='height',
                    name='brush{}'.format(depth),
                    eps=0)
            brushes.append(brush)
            lays.append(brush)
            return
        
        nb_comp_next = nb_comp[depth + 1]
        nb_dim_next = dim_comp[depth + 1]
            
        default_params = (init.GlorotUniform(), init.Constant(0.), init.GlorotUniform(), init.Constant(0.))
        Wcoord, bcoord, Wrepr, brepr = params_per_depth.get(depth, default_params)
        print(params_per_depth)

        for i in range(nb_comp_cur):
            lcoord_cur = layers.SliceLayer(lcoord, i, axis=1, name='coord_cur_{}_{}'.format(i, depth))
            lrepr_cur = layers.SliceLayer(lrepr, i, axis=1, name='repr_cur_{}_{}'.format(i, depth))
            lfeats = layers.ConcatLayer((lcoord_cur, lrepr_cur), axis=1)
            lfeats = layers.DenseLayer(lfeats, 256, nonlinearity=rectify)
            # coord next
            lcoord_next = layers.DenseLayer(
                    lfeats, nb_comp_next * 2, 
                    W=Wcoord, b=bcoord, 
                    nonlinearity=linear, 
                    name='coord_{}_{}'.format(i, depth))
            Wcoord = lcoord_next.W
            bcoord = lcoord_next.b

            lays.append(lcoord_next)
        
            #lcoord_next = batch_norm(lcoord_next)

            lcoord_next = layers.NonlinearityLayer(lcoord_next, T.tanh)
            lcoord_next = layers.ReshapeLayer(lcoord_next, ([0], nb_comp_next, 2))

            def fn(abs, rel):
                abs = abs[:, None, :]
                #r = (abs + abs + scales[depth]) * rel
                #r = abs + abs * rel
                r = abs + rel
                r = theano.tensor.clip(r, 0, 1)
                return r
            lcoord_next = ExpressionLayerMulti(
                    (lcoord_cur, lcoord_next), 
                    fn, 
                    output_shape=lcoord_next.output_shape, 
                    name='coord_{}_{}_normalized'.format(i, depth))
            lays.append(lcoord_next)

            # repr next
            lrepr_next = layers.DenseLayer(
                lrepr_cur, 
                nb_comp_next * nb_dim_next, 
                W=Wrepr, b=brepr, 
                nonlinearity=linear, 
                name='repr_{}_{}'.format(i, depth))
            lays.append(lrepr_next)
            Wrepr = lrepr_next.W
            brepr = lrepr_next.b
            params_per_depth[depth] = Wcoord, bcoord, Wrepr, brepr
            
            #lrepr_next = batch_norm(lrepr_next)
            lrepr_next = layers.ReshapeLayer(lrepr_next, ([0], nb_comp_next, nb_dim_next))
            lrepr_next = layers.ExpressionLayer(lrepr_next, lambda x:selector(x), output_shape='auto', name='repr_{}_{}_normalized'.format(i, depth))
            lays.append(lrepr_next)

            add_program_layer(lrepr_next, lcoord_next, depth=depth + 1)

    add_program_layer(depth=0)
    raw_out = layers.ElemwiseSumLayer(brushes, name="raw_out")
    scaled_out = layers.ScaleLayer(
       raw_out, scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(
      scaled_out, b=init.Constant(-1), name="biased_output")
    out = layers.NonlinearityLayer(
       biased_out,
       nonlinearity=T.nnet.sigmoid,
       name="output")
    return layers_from_list_to_dict([in_] + lays + [raw_out, scaled_out, biased_out, out])


def model97(w=32, h=32, c=1,
            nb_fc_layers=3,
            nb_recurrent_units=100,
            nb_fc_units=1000,
            nb_conv_layers=0,
            nb_conv_filters=64,
            size_conv_filters=3,
            nonlin='relu',
            nonlin_out='sigmoid',
            pooling=True,
            n_steps=10,
            patch_size=3,
            w_out=-1,
            h_out=-1,
            reduce_func='sum',
            proba_func='sparsemax',
            normalize_func='sigmoid',
            update_in_func='prev',
            update_out_func='new',
            x_sigma=0.5,
            y_sigma=0.5,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            nb_patches=1,
            learn_patches=False,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            patches=None,
            merge_op='sum',
            col=None,
            eps=0):

    """
    model88 but using feedback and attention + learn patches
    """

    # INIT
    merge_op = {'sum': lambda x,y:0.5*(x+y), 'mul': lambda x,y: (x*y)}[merge_op]
    def init_method(): return init.GlorotUniform(gain='relu')
    if type(nb_fc_units) != list: nb_fc_units = [nb_fc_units] * nb_fc_layers
    if type(nb_conv_filters) != list: nb_conv_filters = [nb_conv_filters] * nb_conv_layers
    if type(size_conv_filters) != list: size_conv_filters = [size_conv_filters] * nb_conv_layers
    if w_out == -1: w_out = w
    if h_out == -1: h_out = h
    output_shape = (None, c, h_out, w_out)
    nonlin = get_nonlinearity[nonlin]
    if patches is None:patches = np.ones((nb_patches, c, patch_size * (h_out/h), patch_size * (w_out/w)));patches = patches.astype(np.float32)
    extra_layers = []

    ##  ENCODING part
    in_ = layers.InputLayer((None, c, w, h), name="input")
    hid = in_
    nets = []
    hids = conv_fc(
      hid, 
      num_filters=nb_conv_filters, 
      size_conv_filters=size_conv_filters, 
      init_method=init_method,
      pooling=pooling,
      nb_fc_units=nb_fc_units,
      nonlin=nonlin,
      names_prefix='')
    hid = hids[-1]

    nb = ( 2 +
          (1 if x_sigma == 'predicted' else 0) +
          (len(x_sigma) if type(x_sigma) == list else 0) + 
          (1 if y_sigma == 'predicted' else 0) +
          (len(y_sigma) if type(y_sigma) == list else 0) + 
          (1 if x_stride == 'predicted' else 0) +
          (len(x_stride) if type(x_stride) == list else 0) + 
          (1 if y_stride == 'predicted' else 0) +
          (len(y_stride) if type(y_stride) == list else 0) + 
          (c if color == 'predicted' else 0) +
          (1 if patch_index == 'predicted' else 0) +
          (nb_patches if patch_index == 'predicted' else 0))
    hid.name = 'in_to_repr'
    in_to_repr = hid
    hidden_state = layers.InputLayer((None, nb_recurrent_units))

    # get coords from hidden state
    hid_to_out = layers.DenseLayer(hidden_state, nb, nonlinearity=linear, name='hid_to_out')
    # predict read attention window from coords
    hid_to_in = layers.DenseLayer(hidden_state, 4, nonlinearity=linear, name='hid_to_in')
    hid_to_in = layers.ReshapeLayer(hid_to_in, ([0], 1, [1]))
    hid_to_in = GenericBrushLayer(
        hid_to_in,
        w_out, h_out,
        n_steps=1,
        patches=np.ones((1, 1, 1, 1)).astype(np.float32),
        col='grayscale',
        return_seq=False,
        reduce_func=reduce_funcs['prev'],
        to_proba_func=proba_funcs[proba_func],
        normalize_func=normalize_funcs[normalize_func],
        x_sigma='predicted',
        y_sigma='predicted',
        x_stride=1,
        y_stride=1,
        patch_index=0,
        color=[1.],
        x_min=0,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        eps=eps,
        learn_patches=False,
        name="read_attention_window")
    # divide by the max in each canvas to avoid numbers between 0 and 1
    # the values in each canvas sum up to 1 but depending on sigma, can be
    # very small, we want values with high brillance to be 'selected'
    # in the attention mechanism
    hid_to_in = layers.ExpressionLayer(hid_to_in, lambda x:x / (x.max(axis=(2, 3), keepdims=True) +1e-10  ))
    hid_to_in = layers.DenseLayer(hidden_state, w*h, nonlinearity=sigmoid)
    hid_to_in = layers.ReshapeLayer(hid_to_in, ([0], w, h), name='hid_to_in')

    extra_layers.append(hid_to_in)
    extra_layers.append(in_to_repr)
    extra_layers.append(hid_to_out)

    def predict_input(xprev, hprev, oprev):
        attention = layers.get_output(hid_to_in, hprev)[:, None, :, :]
        #return attention * xprev
        return xprev + attention - attention
    
    def predict_repr(x):
        return layers.get_output(in_to_repr, x)

    def predict_output(oprev, hcur):
        return layers.get_output(hid_to_out, hcur)
    
    repr_shape = in_to_repr.output_shape
    out_shape = hid_to_out.output_shape
    
    coord = FeedbackGRULayerClean(
        in_, nb_recurrent_units, 
        predict_input=predict_input,
        predict_output=predict_output,
        predict_repr=predict_repr,
        repr_shape=repr_shape,
        out_shape=out_shape,
        n_steps=n_steps,
        name='coord')

    # DECODING PART
    brush = GenericBrushLayer(
        coord,
        w_out, h_out,
        n_steps=n_steps,
        patches=patches,
        col=col if col else ('rgb' if c == 3 else 'grayscale'),
        return_seq=True,
        reduce_func=reduce_funcs['prev'],
        to_proba_func=proba_funcs[proba_func],
        normalize_func=normalize_funcs[normalize_func],
        x_sigma=x_sigma,
        y_sigma=y_sigma,
        x_stride=x_stride,
        y_stride=y_stride,
        patch_index=patch_index,
        color=color,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        eps=eps,
        learn_patches=learn_patches,
        name="brush"
    )
    def acc(x):
        init_val = T.zeros((x.shape[0],) + (c, h, w))
        init_val = T.unbroadcast(init_val, 0, 1, 2, 3)
        x = x.dimshuffle(1, 0, 2, 3, 4) # time should be first dim
        x, _ = recurrent_accumulation(
            x, 
            apply_func=lambda x:x, 
            reduce_func=reduce_funcs[reduce_func],
            init_val=init_val, 
            n_steps=n_steps)
        x = x.dimshuffle(1, 0, 2, 3, 4) 
        x = x[:, -1]
        return x
    raw_out = layers.ExpressionLayer(
        brush,
        lambda x: acc(x), 
        name="raw_output",
        output_shape='auto')

    raw_out = AddParams(raw_out, [in_to_repr, hid_to_out, hid_to_in], name="raw_output")
    scaled_out = layers.ScaleLayer(
        raw_out, scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(
        scaled_out, b=init.Constant(-1), name="biased_output")
    resid = raw_out
    nfilters = [32, 64, 32, c]
    fs = 3
    i = 0
    for nf in nfilters:
        resid = layers.Conv2DLayer(
                resid,
                num_filters=nf,
                filter_size=(fs, fs),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                pad=(fs-1)/2,
                name="resid_conv{}".format(i))
        extra_layers.append(resid)
        i += 2
    raw_resid_out = resid
    raw_resid_out.name = 'raw_resid_output'
    resid_out = layers.ScaleLayer(raw_resid_out, name='scaled_resid_output')
    resid_out = layers.BiasLayer(resid_out, name="biased_resid_output")

    out = ExpressionLayerMulti((biased_out, resid_out), merge_op)

    out = layers.NonlinearityLayer(
        out,
        nonlinearity=get_nonlinearity[nonlin_out],
        name="output")
    all_layers = ([in_] +
                  hids + extra_layers + [coord, brush] + [raw_out, raw_resid_out, resid_out, scaled_out, biased_out, out])
    return layers_from_list_to_dict(all_layers)

def model98(w=32, h=32, c=1,
            nb_fc_layers=3,
            nb_recurrent_units=100,
            nb_fc_units=1000,
            nb_conv_layers=0,
            nb_conv_filters=64,
            size_conv_filters=3,
            nonlin='relu',
            nonlin_out='sigmoid',
            pooling=True,
            n_steps=10,
            patch_size=3,
            w_out=-1,
            h_out=-1,
            reduce_func='sum',
            proba_func='sparsemax',
            normalize_func='sigmoid',
            update_in_func='prev',
            update_out_func='new',
            x_sigma=0.5,
            y_sigma=0.5,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            nb_patches=1,
            learn_patches=False,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            patches=None,
            col=None,
            eps=0):

    """
    model88 but using feedback and attention + learn patches
    """

    # INIT
    def init_method(): return init.GlorotUniform(gain='relu')
    if type(nb_fc_units) != list: nb_fc_units = [nb_fc_units] * nb_fc_layers
    if type(nb_conv_filters) != list: nb_conv_filters = [nb_conv_filters] * nb_conv_layers
    if type(size_conv_filters) != list: size_conv_filters = [size_conv_filters] * nb_conv_layers
    if w_out == -1: w_out = w
    if h_out == -1: h_out = h
    output_shape = (None, c, h_out, w_out)
    nonlin = get_nonlinearity[nonlin]
    if patches is None:patches = np.ones((nb_patches, c, patch_size * (h_out/h), patch_size * (w_out/w)));patches = patches.astype(np.float32)
    extra_layers = []

    ##  ENCODING part
    in_ = layers.InputLayer((None, c, w, h), name="input")
    hid = in_
    nets = []
    hids = conv_fc(
      hid, 
      num_filters=nb_conv_filters, 
      size_conv_filters=size_conv_filters, 
      init_method=init_method,
      pooling=pooling,
      nb_fc_units=nb_fc_units,
      nonlin=nonlin,
      names_prefix='')
    hid = hids[-1]

    nb = ( 2 +
          (1 if x_sigma == 'predicted' else 0) +
          (len(x_sigma) if type(x_sigma) == list else 0) + 
          (1 if y_sigma == 'predicted' else 0) +
          (len(y_sigma) if type(y_sigma) == list else 0) + 
          (1 if x_stride == 'predicted' else 0) +
          (len(x_stride) if type(x_stride) == list else 0) + 
          (1 if y_stride == 'predicted' else 0) +
          (len(y_stride) if type(y_stride) == list else 0) + 
          (c if color == 'predicted' else 0) +
          (1 if patch_index == 'predicted' else 0) +
          (nb_patches if patch_index == 'predicted' else 0))
    hid.name = 'in_to_repr'
    in_to_repr = hid
    hidden_state = layers.InputLayer((None, nb_recurrent_units))

    # get coords from hidden state
    hid_to_out = layers.DenseLayer(hidden_state, nb, nonlinearity=linear, name='hid_to_out')
    # predict read attention window from coords
    hid_to_in = layers.DenseLayer(hidden_state, 4, nonlinearity=linear, name='hid_to_in')
    hid_to_in = layers.ReshapeLayer(hid_to_in, ([0], 1, [1]))
    hid_to_in = GenericBrushLayer(
        hid_to_in,
        w_out, h_out,
        n_steps=1,
        patches=np.ones((1, 1, 1, 1)).astype(np.float32),
        col='grayscale',
        return_seq=False,
        reduce_func=reduce_funcs['prev'],
        to_proba_func=proba_funcs[proba_func],
        normalize_func=normalize_funcs[normalize_func],
        x_sigma='predicted',
        y_sigma='predicted',
        x_stride=1,
        y_stride=1,
        patch_index=0,
        color=[1.],
        x_min=0,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        eps=eps,
        learn_patches=False,
        name="read_attention_window")
    # divide by the max in each canvas to avoid numbers between 0 and 1
    # the values in each canvas sum up to 1 but depending on sigma, can be
    # very small, we want values with high brillance to be 'selected'
    # in the attention mechanism
    hid_to_in = layers.ExpressionLayer(hid_to_in, lambda x:x / (x.max(axis=(2, 3), keepdims=True) +1e-10  ))
    hid_to_in = layers.DenseLayer(hidden_state, w*h, nonlinearity=sigmoid)
    hid_to_in = layers.ReshapeLayer(hid_to_in, ([0], w, h), name='hid_to_in')

    extra_layers.append(hid_to_in)
    extra_layers.append(in_to_repr)
    extra_layers.append(hid_to_out)

    def predict_input(xprev, hprev, oprev):
        attention = layers.get_output(hid_to_in, hprev)[:, None, :, :]
        #return attention * xprev
        return xprev + attention - attention
    
    def predict_repr(x):
        return layers.get_output(in_to_repr, x)

    def predict_output(oprev, hcur):
        return layers.get_output(hid_to_out, hcur)
    
    repr_shape = in_to_repr.output_shape
    out_shape = hid_to_out.output_shape
    
    coord = FeedbackGRULayerClean(
        in_, nb_recurrent_units, 
        predict_input=predict_input,
        predict_output=predict_output,
        predict_repr=predict_repr,
        repr_shape=repr_shape,
        out_shape=out_shape,
        n_steps=n_steps,
        name='coord')

    # DECODING PART
    brush = GenericBrushLayer(
        coord,
        w_out, h_out,
        n_steps=n_steps,
        patches=patches,
        col=col if col else ('rgb' if c == 3 else 'grayscale'),
        return_seq=True,
        reduce_func=reduce_funcs['prev'],
        to_proba_func=proba_funcs[proba_func],
        normalize_func=normalize_funcs[normalize_func],
        x_sigma=x_sigma,
        y_sigma=y_sigma,
        x_stride=x_stride,
        y_stride=y_stride,
        patch_index=patch_index,
        color=color,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        eps=eps,
        learn_patches=learn_patches,
        name="brush"
    )
    def acc(x):
        init_val = T.zeros((x.shape[0],) + (c, h, w))
        init_val = T.unbroadcast(init_val, 0, 1, 2, 3)
        x = x.dimshuffle(1, 0, 2, 3, 4) # time should be first dim
        x, _ = recurrent_accumulation(
            x, 
            apply_func=lambda x:x, 
            reduce_func=reduce_funcs[reduce_func],
            init_val=init_val, 
            n_steps=n_steps)
        x = x.dimshuffle(1, 0, 2, 3, 4) 
        x = x[:, -1]
        return x
    raw_out = layers.ExpressionLayer(
        brush,
        lambda x: acc(x), 
        name="raw_output",
        output_shape='auto')
    raw_out = AddParams(raw_out, [in_to_repr, hid_to_out, hid_to_in], name="raw_output")
    resid = raw_out
    nfilters = [32, 64, 32, c]
    fs = 3
    i = 0
    for nf in nfilters:
        resid = layers.Conv2DLayer(
                resid,
                num_filters=nf,
                filter_size=(fs, fs),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                pad=(fs-1)/2,
                name="resid_conv{}".format(i))
        extra_layers.append(resid)
        i += 1
    raw_resid_out = resid
    raw_resid_out.name = 'raw_resid_output'
    scaled_out = layers.ScaleLayer(raw_resid_out, name='scaled_output')
    out = layers.NonlinearityLayer(scaled_out, nonlinearity=get_nonlinearity[nonlin_out], name='output')
    all_layers = ([in_] +
                  hids + extra_layers + [coord, brush] + [raw_out, raw_resid_out, scaled_out, out])
    return layers_from_list_to_dict(all_layers)

def model99(w=32, h=32, c=1,
            nb_fc_layers=3,
            nb_recurrent_units=100,
            nb_fc_units=1000,
            nb_conv_layers=0,
            nb_conv_filters=64,
            size_conv_filters=3,
            nonlin='relu',
            nonlin_out='sigmoid',
            pooling=True,
            n_steps=10,
            patch_size=3,
            w_out=-1,
            h_out=-1,
            reduce_func='sum',
            proba_func='sparsemax',
            normalize_func='sigmoid',
            update_in_func='prev',
            update_out_func='new',
            x_sigma=0.5,
            y_sigma=0.5,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            nb_patches=1,
            learn_patches=False,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            patches=None,
            merge_op='sum',
            col=None,
            eps=0):

    """
    model88 but using feedback and attention + learn patches
    """

    # INIT
    merge_op = {'sum': lambda x,y:0.5*(x+y), 'mul': lambda x,y: (x*y)}[merge_op]
    def init_method(): return init.GlorotUniform(gain='relu')
    if type(nb_fc_units) != list: nb_fc_units = [nb_fc_units] * nb_fc_layers
    if type(nb_conv_filters) != list: nb_conv_filters = [nb_conv_filters] * nb_conv_layers
    if type(size_conv_filters) != list: size_conv_filters = [size_conv_filters] * nb_conv_layers
    if w_out == -1: w_out = w
    if h_out == -1: h_out = h
    output_shape = (None, c, h_out, w_out)
    nonlin = get_nonlinearity[nonlin]
    if patches is None:patches = np.ones((nb_patches, c, patch_size * (h_out/h), patch_size * (w_out/w)));patches = patches.astype(np.float32)
    extra_layers = []

    ##  ENCODING part
    in_ = layers.InputLayer((None, c, w, h), name="input")
    hid = in_
    nets = []
    hids = conv_fc(
      hid, 
      num_filters=nb_conv_filters, 
      size_conv_filters=size_conv_filters, 
      init_method=init_method,
      pooling=pooling,
      nb_fc_units=nb_fc_units,
      nonlin=nonlin,
      names_prefix='')
    hid = hids[-1]

    nb = ( 2 +
          (1 if x_sigma == 'predicted' else 0) +
          (len(x_sigma) if type(x_sigma) == list else 0) + 
          (1 if y_sigma == 'predicted' else 0) +
          (len(y_sigma) if type(y_sigma) == list else 0) + 
          (1 if x_stride == 'predicted' else 0) +
          (len(x_stride) if type(x_stride) == list else 0) + 
          (1 if y_stride == 'predicted' else 0) +
          (len(y_stride) if type(y_stride) == list else 0) + 
          (c if color == 'predicted' else 0) +
          (1 if patch_index == 'predicted' else 0) +
          (nb_patches if patch_index == 'predicted' else 0))
    hid.name = 'in_to_repr'
    in_to_repr = hid
    hidden_state = layers.InputLayer((None, nb_recurrent_units))

    # get coords from hidden state
    hid_to_out = layers.DenseLayer(hidden_state, nb, nonlinearity=linear, name='hid_to_out')
    # predict read attention window from coords
    hid_to_in = layers.DenseLayer(hidden_state, 4, nonlinearity=linear, name='hid_to_in')
    hid_to_in = layers.ReshapeLayer(hid_to_in, ([0], 1, [1]))
    hid_to_in = GenericBrushLayer(
        hid_to_in,
        w_out, h_out,
        n_steps=1,
        patches=np.ones((1, 1, 1, 1)).astype(np.float32),
        col='grayscale',
        return_seq=False,
        reduce_func=reduce_funcs['prev'],
        to_proba_func=proba_funcs[proba_func],
        normalize_func=normalize_funcs[normalize_func],
        x_sigma='predicted',
        y_sigma='predicted',
        x_stride=1,
        y_stride=1,
        patch_index=0,
        color=[1.],
        x_min=0,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        eps=eps,
        learn_patches=False,
        name="read_attention_window")
    # divide by the max in each canvas to avoid numbers between 0 and 1
    # the values in each canvas sum up to 1 but depending on sigma, can be
    # very small, we want values with high brillance to be 'selected'
    # in the attention mechanism
    hid_to_in = layers.ExpressionLayer(hid_to_in, lambda x:x / (x.max(axis=(2, 3), keepdims=True) +1e-10  ))
    hid_to_in = layers.DenseLayer(hidden_state, w*h, nonlinearity=sigmoid)
    hid_to_in = layers.ReshapeLayer(hid_to_in, ([0], w, h), name='hid_to_in')

    extra_layers.append(hid_to_in)
    extra_layers.append(in_to_repr)
    extra_layers.append(hid_to_out)

    def predict_input(xprev, hprev, oprev):
        attention = layers.get_output(hid_to_in, hprev)[:, None, :, :]
        #return attention * xprev
        return xprev + attention - attention
    
    def predict_repr(x):
        return layers.get_output(in_to_repr, x)

    def predict_output(oprev, hcur):
        return layers.get_output(hid_to_out, hcur)
    
    repr_shape = in_to_repr.output_shape
    out_shape = hid_to_out.output_shape
    
    coord = FeedbackGRULayerClean(
        in_, nb_recurrent_units, 
        predict_input=predict_input,
        predict_output=predict_output,
        predict_repr=predict_repr,
        repr_shape=repr_shape,
        out_shape=out_shape,
        n_steps=n_steps,
        name='coord')

    # DECODING PART
    brush = GenericBrushLayer(
        coord,
        w_out, h_out,
        n_steps=n_steps,
        patches=patches,
        col=col if col else ('rgb' if c == 3 else 'grayscale'),
        return_seq=True,
        reduce_func=reduce_funcs['prev'],
        to_proba_func=proba_funcs[proba_func],
        normalize_func=normalize_funcs[normalize_func],
        x_sigma=x_sigma,
        y_sigma=y_sigma,
        x_stride=x_stride,
        y_stride=y_stride,
        patch_index=patch_index,
        color=color,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        eps=eps,
        learn_patches=learn_patches,
        name="brush"
    )
    def acc(x):
        init_val = T.zeros((x.shape[0],) + (c, h, w))
        init_val = T.unbroadcast(init_val, 0, 1, 2, 3)
        x = x.dimshuffle(1, 0, 2, 3, 4) # time should be first dim
        x, _ = recurrent_accumulation(
            x, 
            apply_func=lambda x:x, 
            reduce_func=reduce_funcs[reduce_func],
            init_val=init_val, 
            n_steps=n_steps)
        x = x.dimshuffle(1, 0, 2, 3, 4) 
        x = x[:, -1]
        return x
    raw_out = layers.ExpressionLayer(
        brush,
        lambda x: acc(x), 
        name="raw_output",
        output_shape='auto')

    raw_out = AddParams(raw_out, [in_to_repr, hid_to_out, hid_to_in], name="raw_output")
    scaled_out = layers.ScaleLayer(
        raw_out, scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(
        scaled_out, b=init.Constant(-1), name="biased_output")
    out = layers.NonlinearityLayer(
        biased_out,
        nonlinearity=get_nonlinearity[nonlin_out],
        name="output")
    all_layers = ([in_] +
                  hids + extra_layers + [coord, brush] + [raw_out, scaled_out, biased_out, out])
    return layers_from_list_to_dict(all_layers)

def model100(w=32, h=32, c=1,
            nb_fc_layers=3,
            nb_recurrent_layers=1,
            nb_recurrent_units=100,
            nb_fc_units=1000,
            nb_conv_layers=0,
            nb_conv_filters=64,
            size_conv_filters=3,
            nonlin='relu',
            nonlin_out='sigmoid',
            pooling=True,
            n_steps=10,
            patch_size=3,
            w_out=-1,
            h_out=-1,
            reduce_func='sum',
            proba_func='sparsemax',
            normalize_func='sigmoid',
            x_sigma=0.5,
            y_sigma=0.5,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            nb_patches=1,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            recurrent_model='gru',
            variational=False,
            variational_nb_hidden=100,
            variational_seed=1,
            patches=None,
            col=None,
            parallel=1,
            parallel_share=True,
            parallel_reduce_func='sum',
            merge_op_resid='mean',
            nb_filters_resid=[],
            size_filters_resid=[],
            nb_levels_resid=1,
            eps=0):

    """
    model88 with enhancement module like in model97
    """

    # INIT
    def init_method(): return init.GlorotUniform(gain='relu')
    merge_op_resid = {'mean': lambda x,y:0.5*(x+y), 'mul': lambda x,y: (x*y), 'resid': lambda x, y:y, 'orig': lambda x,y:x}[merge_op_resid]
    if type(nb_fc_units) != list: nb_fc_units = [nb_fc_units] * nb_fc_layers
    if type(nb_conv_filters) != list: nb_conv_filters = [nb_conv_filters] * nb_conv_layers
    if type(size_conv_filters) != list: size_conv_filters = [size_conv_filters] * nb_conv_layers
    if type(nb_recurrent_units) != list: nb_recurrent_units = [nb_recurrent_units] * nb_recurrent_layers
    if w_out == -1: w_out = w
    if h_out == -1: h_out = h
    output_shape = (None, c, h_out, w_out)
    nonlin = get_nonlinearity[nonlin]
    recurrent_model = recurrent_models[recurrent_model]
    if patches is None:patches = np.ones((1, c, patch_size * (h_out/h), patch_size * (w_out/w)));patches = patches.astype(np.float32)
    extra_layers = []

    ##  ENCODING part
    in_ = layers.InputLayer((None, c, w, h), name="input")
    hid = in_
        
    nets = []
    for i in range(parallel):
        hids = conv_fc(
          hid, 
          num_filters=nb_conv_filters, 
          size_conv_filters=size_conv_filters, 
          init_method=init_method,
          pooling=pooling,
          nb_fc_units=nb_fc_units,
          nonlin=nonlin,
          names_prefix='net{}'.format(i))
        nets.append(hids)
   
    if variational:
        all_z_mu = []
        all_z_log_sigma = []
        for n, net in enumerate(nets):
            z_mu = layers.DenseLayer(net[-1], variational_nb_hidden, nonlinearity=linear, name='z_mu_{}'.format(n))
            z_log_sigma = layers.DenseLayer(net[-1], variational_nb_hidden, nonlinearity=linear, name='z_log_sigma_{}'.format(n))
            all_z_mu.append(z_mu)
            all_z_log_sigma.append(z_log_sigma)
            net.append(z_mu)
            net.append(z_log_sigma)

        z_mu = layers.ConcatLayer(all_z_mu, axis=1, name='z_mu')
        z_log_sigma = layers.ConcatLayer(all_z_log_sigma, axis=1, name='z_log_sigma')
        z = GaussianSampleLayer(
            z_mu, z_log_sigma,
            rng=RandomStreams(variational_seed),
            name='z_sample')

        extra_layers.append(z_mu)
        extra_layers.append(z_log_sigma)
        extra_layers.append(z)
        i = 0
        for n, net in enumerate(nets):
            z_net = layers.SliceLayer(z, indices=slice(i, i + variational_nb_hidden), axis=1, name='z_sample_{}'.format(n))
            i += variational_nb_hidden
            net.append(z_net)

    for n, net in enumerate(nets):
        hid = Repeat(net[-1], n_steps)
        for l in range(nb_recurrent_layers):
            hid = recurrent_model(
                    hid, 
                    nb_recurrent_units[l], 
                    name="recurrent{}_{}".format(l, n))
        net.append(hid)
    
    nb = ( 2 +
          (1 if x_sigma == 'predicted' else 0) +
          (len(x_sigma) if type(x_sigma) == list else 0) + 
          (1 if y_sigma == 'predicted' else 0) +
          (len(y_sigma) if type(y_sigma) == list else 0) + 
          (1 if x_stride == 'predicted' else 0) +
          (len(x_stride) if type(x_stride) == list else 0) + 
          (1 if y_stride == 'predicted' else 0) +
          (len(y_stride) if type(y_stride) == list else 0) + 
          (c if color == 'predicted' else 0) +
          (1 if patch_index == 'predicted' else 0) +
          (nb_patches if patch_index == 'predicted' else 0))
    for i, net in enumerate(nets):
        hid = net[-1]
        coord = TensorDenseLayer(hid, nb, nonlinearity=linear, name="coord_{}".format(i))
        net.append(coord)

    # DECODING PART
    learn_patches = (patch_index == 'predicted')
    for i, net in enumerate(nets):
        coord = net[-1]
        brush = GenericBrushLayer(
            coord,
            w_out, h_out,
            n_steps=n_steps,
            patches=patches,
            col=col if col else ('rgb' if c == 3 else 'grayscale'),
            return_seq=True,
            reduce_func=reduce_funcs[reduce_func],
            to_proba_func=proba_funcs[proba_func],
            normalize_func=normalize_funcs[normalize_func],
            x_sigma=x_sigma,
            y_sigma=y_sigma,
            x_stride=x_stride,
            y_stride=y_stride,
            patch_index=patch_index,
            color=color,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            learn_patches=learn_patches,
            eps=eps,
            name="brush_{}".format(i)
        )
        net.append(brush)

    for i, net in enumerate(nets):
        brush = net[-1]
        raw_out = layers.ExpressionLayer(
            brush,
            lambda x: x[:, -1, :, :],
            name="raw_output_{}".format(i),
            output_shape='auto')
        net.append(brush)
    
    def nets_reduce(*nets):
        nets = map(lambda k:k[:, -1], nets) # get last time step output from each net
        func = reduce_funcs[parallel_reduce_func]
        return reduce(func, nets)

    raw_out = ExpressionLayerMulti(
        map(lambda net:net[-1], nets),
        nets_reduce,
        name="raw_output",
        output_shape=output_shape)
    
    scaled_out = layers.ScaleLayer(
        raw_out, scales=init.Constant(2.), name="scaled_output")
    biased_out = layers.BiasLayer(
        scaled_out, b=init.Constant(-1), name="biased_output")

    resid = raw_out
    nb_filters_resid = nb_filters_resid
    levels = []
    for i in range(len(nb_filters_resid)):
        resid = layers.Conv2DLayer(
                resid,
                num_filters=nb_filters_resid[i],
                filter_size=(size_filters_resid[i], size_filters_resid[i]),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                pad=(size_filters_resid[i]-1)/2,
                name="resid_conv{}".format(i))
        resid_out = layers.Conv2DLayer(
                resid,
                num_filters=c,
                filter_size=(size_filters_resid[i], size_filters_resid[i]),
                nonlinearity=rectify,
                W=init.GlorotUniform(),
                pad=(size_filters_resid[i]-1)/2,
                name="resid_conv{}".format(i))
        levels.append(resid_out)
    
    levels = levels[::-1][0:nb_levels_resid]
    print(levels)
    def mean_op(*args):
        return sum(args) / len(args)
    raw_resid_out = ExpressionLayerMulti(levels, mean_op)
    raw_resid_out.name = 'raw_resid_output'

    resid_out = layers.ScaleLayer(raw_resid_out, name='scaled_resid_output')
    resid_out = layers.BiasLayer(resid_out, name="biased_resid_output")
    out = ExpressionLayerMulti((biased_out, resid_out), merge_op_resid)
    out = layers.NonlinearityLayer(
        out,
        nonlinearity=get_nonlinearity[nonlin_out],
        name="output")
    all_layers = ([in_] + 
                  [lay for net in nets for lay in net] + 
                  extra_layers + 
                  [raw_out, raw_resid_out, resid_out, scaled_out, biased_out, out])
    return layers_from_list_to_dict(all_layers)

def conv_fc(x, 
            num_filters=[32, 32], 
            size_conv_filters=[5, 5], 
            init_method=init.GlorotUniform,
            pooling=False,
            nb_fc_units=[100],
            nonlin=get_nonlinearity['rectify'],
            names_prefix=''):
    l_hid = x
    hids = []
    for i in range(len(num_filters)):
        l_hid = layers.Conv2DLayer(
            l_hid,
            num_filters=num_filters[i],
            filter_size=(size_conv_filters[i], size_conv_filters[i]),
            nonlinearity=nonlin,
            W=init_method(),
            name='conv{}_{}'.format(i + 1, names_prefix))
        hids.append(l_hid)
        if pooling:
            l_hid = layers.Pool2DLayer(l_hid, (2, 2), name='pool{}_{}'.format(i + 1, names_prefix))
            hids.append(l_hid)

    for i in range(len(nb_fc_units)):
        l_hid = layers.DenseLayer(
            l_hid, nb_fc_units[i],
            W=init_method(),
            nonlinearity=nonlin,
            name="fc{}_{}".format(i + 1, names_prefix))
        hids.append(l_hid)
    return hids


def model101(nb_filters=64, w=32, h=32, c=1,
            nb_layers=None,
            filter_size=5,
            sparse_func='wta_spatial',
            k=1,
            weight_sharing=False,
            merge_op='sum'):
    """
    model73 but with different kinds of sparsity
    """
    merge_op = {'sum': T.add, 'mul': T.mul, 'over': over_op}[merge_op]
    sparse_funcs  ={
        'wta_k_spatial': wta_k_spatial,
        'wta_spatial': lambda k:wta_spatial,
        'max_k_spatial': max_k_spatial,
    }
    if nb_layers is None:
        nb_layers = len(nb_filters)
    if type(filter_size) != list:
        filter_size = [filter_size] * nb_layers
    if type(nb_filters) != list:
        nb_filters = [nb_filters] * nb_layers
    if type(k) != list:
        k = [k] * nb_layers
    if type(weight_sharing) != list:
        weight_sharing = [weight_sharing] * nb_layers
    sparse_layers = []
    def sparse(l):
        i = len(sparse_layers)
        fn = sparse_funcs[sparse_func](nb=k[i])
        l = layers.NonlinearityLayer(l, fn, name='sparse{}'.format(i))
        sparse_layers.append(l)
        return l

    l_in = layers.InputLayer((None, c, w, h), name="input")
    l_conv = l_in
    convs = []
    convs_sparse = []
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters[i],
            filter_size=(filter_size[i], filter_size[i]),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv{}".format(i + 1))
        print(l_conv.output_shape)
        convs.append(l_conv)
        l_conv_sparse = sparse(l_conv)
        convs_sparse.append(l_conv_sparse)

    conv_backs = []
    back = {}
    back_layers = {}
    for i in range(nb_layers): #[0, 1, 2]
        l_conv_back = convs_sparse[i]
        for j in range(i): # for 0 : [], for 1 : [0], for 2 : [0, 1]
            if weight_sharing[i - j - 1] and i > 0 and j > 0:
                W = back_layers[(i - 1, j - 1)].W
            else:
                W = init.GlorotUniform()
            l_conv_back = layers.Conv2DLayer(
                l_conv_back,
                num_filters=nb_filters[i - j - 1],
                filter_size=(filter_size[i - j - 1], filter_size[i - j - 1]),
                nonlinearity=rectify,
                W=W,
                pad='full',
                name='conv_back_{}_{}'.format(i + 1, j + 1)
            )
            back_layers[(i, j)] = l_conv_back
            #back[(i, j)] = l_conv_back.W
        #l_conv_back.name = 'conv_back{}'.format(i + 1)
        conv_backs.append(l_conv_back)
    #print(conv_backs)
    outs = []
    for i, conv_back in enumerate(conv_backs):
        if i == 0 or not weight_sharing[0]:
            W = init.GlorotUniform()
        else:
            W = outs[0].W
        l_out = layers.Conv2DLayer(
            conv_back,
            num_filters=c,
            filter_size=(filter_size[0], filter_size[0]),
            nonlinearity=linear,
            W=W,
            pad='full',
            name='out{}'.format(i + 1))
        outs.append(l_out)
    l_out = layers.ElemwiseMergeLayer(outs, merge_op)
    l_out = layers.NonlinearityLayer(l_out, sigmoid, name='output')
    all_layers = [l_in] + convs + sparse_layers + back_layers.values() + outs + [l_out]
    return layers_from_list_to_dict(all_layers)


def model102(w=32, h=32, c=1, n_steps=1, patch_size=4):
    """
    """
    l_in = layers.InputLayer((None, c, w, h), name="input")
    hid = layers.Conv2DLayer(
            l_in,
            num_filters=64,
            filter_size=(5, 5),
            nonlinearity=rectify,
            W=init.GlorotUniform(),
            name="conv")
    #hid = layers.DenseLayer(l_in, 256, nonlinearity=rectify, name="hid")
    hid = layers.DenseLayer(hid, 128, nonlinearity=rectify, name="hid")

    #hid = layers.DenseLayer(hid, n_steps * 2, nonlinearity=linear, name="hid")
    #l_coord = layers.ReshapeLayer(hid, ([0], n_steps, 2), name="coord")
    hid = Repeat(hid, n_steps)
    #hid = layers.DenseLayer(hid, 256*n_steps, nonlinearity=rectify, name="hid")
    #hid = layers.ReshapeLayer(hid, ([0], n_steps, 256), name="hid")
    hid = layers.GRULayer(hid, 256)
    l_coord = TensorDenseLayer(hid, 100, nonlinearity=linear, name="coord")
    patches = np.ones((1, c, patch_size, patch_size))
    patches = patches.astype(np.float32)
    l_brush = GenericBrushLayer(
        l_coord,
        w, h,
        n_steps=n_steps,
        patches=patches,
        col='rgb' if c == 3 else 'grayscale',
        return_seq=False,
        reduce_func=reduce_funcs['sum'],
        to_proba_func=proba_funcs['softmax'],
        normalize_func=normalize_funcs['sigmoid'],
        x_sigma=0.5,
        y_sigma=0.5,
        x_stride=[0.25, 1],
        y_stride=[0.25, 1],
        patch_index=0,
        color=np.ones((18, 3)).astype(np.float32),
        x_min=-8,
        x_max=24,
        y_min=-8,
        y_max=24,
        h_left_pad=16,
        h_right_pad=16,
        w_left_pad=16,
        w_right_pad=16,
        color_min=-1,
        color_max=1,
        name="brush",
        coords='continuous',
    )
    l_raw_out = l_brush
    l_out = l_raw_out
    l_scaled_out = l_raw_out
    l_biased_out = l_biased_out
    #l_scaled_out = layers.ScaleLayer(
    #    l_raw_out, scales=init.Constant(2.), name="scaled_output")
    #l_biased_out = layers.BiasLayer(
    #    l_scaled_out, b=init.Constant(-1), name="biased_output")
    
    #l_out = layers.NonlinearityLayer(
    #    l_biased_out,
    #    nonlinearity=get_nonlinearity['sigmoid'],
    #    name="output")
    all_layers = ([l_in] +
                  [l_coord, l_brush, l_raw_out, l_biased_out, l_scaled_out,  l_out])
    return layers_from_list_to_dict(all_layers)

build_convnet_simple = model1
build_convnet_simple_2 = model2
build_convnet_simple_3 = model3
build_convnet_simple_4 = model4
build_convnet_simple_5 = model5
build_convnet_simple_6 = model6
build_convnet_simple_7 = model7
build_convnet_simple_8 = model8
build_convnet_simple_9 = model9
build_convnet_simple_10 = model10
build_convnet_simple_11 = model12
build_convnet_simple_12 = model12
build_convnet_simple_13 = model13
build_convnet_simple_14 = model14
build_convnet_simple_15 = model15
build_convnet_simple_16 = model16

if __name__ == '__main__':
    from docopt import docopt
    import theano
    import theano.tensor as T
    import lasagne
    from lasagnekit.misc.draw_net import  draw_to_file

    doc = """
    Usage: model.py MODEL
    """
    args = docopt(doc)
    model = args['MODEL']
    model = globals()[model]
    w, h, c = 28, 28, 1
    all_layers = model(w=w, h=h, c=c)
    draw_to_file(lasagne.layers.get_all_layers(all_layers['output']), 'out.svg')
    for layer in all_layers.items():
        print(layer)
    x = T.tensor4()
    fn = theano.function([x], lasagne.layers.get_output(all_layers['output'], x))
    x_examples = np.random.uniform(size=(128, c, h, w)).astype(np.float32)
    print(fn(x_examples).shape)
