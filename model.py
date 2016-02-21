from lasagne import layers, init
from lasagnekit.easy import layers_from_list_to_dict
from lasagne.nonlinearities import (
        linear, sigmoid, rectify, very_leaky_rectify)
from lasagnekit.layers import Deconv2DLayer, Depool2DLayer
from helpers import wta_spatial, wta_lifetime, wta_channel, wta_channel_strided
import theano.tensor as T


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
