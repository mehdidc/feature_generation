from lasagne import layers, init
from lasagnekit.easy import layers_from_list_to_dict
from lasagne.nonlinearities import (
        linear, sigmoid, rectify, very_leaky_rectify, softmax, tanh)
from lasagnekit.layers import Deconv2DLayer
from helpers import FeedbackGRULayer, TensorDenseLayer
from helpers import Deconv2DLayer as deconv2d
from helpers import correct_over_op, over_op, sum_op, max_op, thresh_op, normalized_over_op
from helpers import wta_spatial, wta_k_spatial, wta_lifetime, wta_channel, wta_channel_strided, wta_fc_lifetime, wta_fc_sparse, norm_maxmin
from helpers import Repeat
from helpers import BrushLayer
import theano.tensor as T
import numpy as np


from batch_norm import NormalizeLayer, ScaleAndShiftLayer, DecoderNormalizeLayer, DenoiseLayer, FakeLayer, SimpleScaleAndShiftLayer
from lasagne.layers import batch_norm

import theano

get_nonlinearity = dict(
    linear=linear,
    sigmoid=sigmoid,
    rectify=rectify,
    very_leaky_rectify=very_leaky_rectify,
    softmax=softmax,
    tanh=tanh
)


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
    like model66 but with sum instead of multiplication
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
    like model67 but with weight sharing in the last layers
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
    like model68 but only two layers
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
    model68 with stride (used for large images)
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
    model70 with sharing
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
    parametrized version of model67
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

    print(weight_sharing)
    if type(weight_sharing) != list:
        weight_sharing = [weight_sharing] * nb_layers
    sparse_layers = []

    print('nb_filters : {}'.format(nb_filters))
    print('nb_layers : {}'.format(nb_layers))

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
                pad='full'
            )
            back[(i, j)] = l_conv_back.W
        l_conv_back.name = 'conv_back{}'.format(i + 1)
        conv_backs.append(l_conv_back)
    print(conv_backs)
    outs = []
    for i, conv_back in enumerate(conv_backs):
        if i == 0 or not weight_sharing[0]:
            W = init.GlorotUniform()
        else:
            W = outs[0].W
            print(W)
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
    all_layers = [l_in] + convs + sparse_layers + conv_backs + outs + [l_out]
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
                   'prev':  lambda a, b: a,
                   'new': lambda a, b: b,
                   'sub': lambda a, b: a - T.nnet.sigmoid(b),
                   'correct_over': correct_over_op(alpha)}

    l_in = layers.InputLayer((None, c, w, h), name="input")
    nonlin = get_nonlinearity[nonlin]

    patch = np.ones((patch_size * (w_out/w), patch_size * (h_out/h)))
    #patch /= np.prod(patch.shape)

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
