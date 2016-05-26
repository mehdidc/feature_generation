from lasagne import layers, init
from lasagnekit.easy import layers_from_list_to_dict
from lasagne.nonlinearities import (
        linear, sigmoid, rectify, very_leaky_rectify, softmax, tanh)
from lasagnekit.layers import Deconv2DLayer, Depool2DLayer
from helpers import wta_spatial, wta_k_spatial, wta_lifetime, wta_channel, wta_channel_strided, wta_fc_lifetime
import theano.tensor as T
import numpy as np
from batch_norm import NormalizeLayer, ScaleAndShiftLayer, DecoderNormalizeLayer, DenoiseLayer, FakeLayer
from lasagne.layers import batch_norm

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
            nb_hidden_units=1000):
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
    l_out = layers.NonlinearityLayer(l_pre_out, sigmoid, name="output")
    l_out = layers.ReshapeLayer(l_out, ([0], c, w, h), name="output")
    print(l_out.output_shape)
    return layers_from_list_to_dict([l_in] + hids + [l_pre_out, l_out])

def model57(nb_filters=64, w=32, h=32, c=1,
            use_wta_lifetime=True,
            wta_lifetime_perc=0.1,
            nb_hidden_units=1000,
            tied=False):
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
    l_out = layers.NonlinearityLayer(l_pre_out, sigmoid, name="output")
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

def model63(nb_filters=64, w=32, h=32, c=1, up=2, filter_size=3, nb_layers=3, sparsity=True):
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
    for i in range(nb_layers):
        l_conv = layers.Conv2DLayer(
            l_conv,
            num_filters=nb_filters,
            filter_size=(filter_size[i + 1], filter_size[i + 1]),
            pad='same', 
            nonlinearity=rectify,
            W=init.GlorotUniform(),
        )

    l_output = layers.Conv2DLayer(
        l_conv,
        num_filters=c,
        filter_size=(filter_size[-1], filter_size[-1]),
        nonlinearity=linear,
        pad='same',
        W=init.GlorotUniform(),
        name="output")
    return layers_from_list_to_dict([l_in, l_output])

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
build_convnet_simple_11 = model11
build_convnet_simple_12 = model12
build_convnet_simple_13 = model13
build_convnet_simple_14 = model14
build_convnet_simple_15 = model15
build_convnet_simple_16 = model16
