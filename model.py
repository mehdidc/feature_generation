from lasagne import layers, init
from lasagnekit.easy import layers_from_list_to_dict
from lasagne.nonlinearities import rectify, linear, sigmoid, tanh
import theano.tensor as T
import numpy as np


def thresh_linear(x):
    return T.maximum(T.minimum(x, 1), 0)


def build_convnet(nb_filters=64, size_filters=5, nb_hidden=1000,
                  w=32, h=32, c=1):
    l_in = layers.InputLayer((None, w*h*c), name="input")
    x_in_reshaped = layers.ReshapeLayer(l_in, ([0], c, w, h), name="input_r")
    l_conv0 = x_in_reshaped

    nonlin = tanh 
    #nonlin = thresh_linear

    l_conv1 = layers.Conv2DLayer(
        l_conv0,
        num_filters=nb_filters,
        filter_size=(size_filters, size_filters),
        nonlinearity=nonlin,
        b=None,
        name="conv1",
    )

    l_conv2 = layers.Conv2DLayer(
        l_conv1,
        num_filters=nb_filters,
        filter_size=(size_filters, size_filters),
        nonlinearity=nonlin,
        b=None,
        name="conv2",
    )

    l_unconv2 = layers.Conv2DLayer(
        l_conv2,
        num_filters=nb_filters,
        filter_size=(size_filters, size_filters),
        nonlinearity=nonlin,
        pad='full',
        name="unconv2",
        W=l_conv2.W.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1],
        b=None
    )

    l_unconv1 = layers.Conv2DLayer(
        l_unconv2,
        num_filters=1,
        filter_size=(size_filters, size_filters),
        nonlinearity=nonlin,
        pad='full',
        name="unconv1",
        W=l_conv1.W.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1],
        b=None
    )

    l_out = layers.ReshapeLayer(
            l_unconv1,
            ([0], c * w * h), name="output")
    #l_out = layers.NonlinearityLayer(
    #        l_out,
    #        nonlinearities.sigmoid, name="output")
    layers_ = [
        l_in,
        l_conv1,
        l_conv2,
        l_unconv2,
        l_unconv1,
        l_out,
    ]
    return layers_from_list_to_dict(layers_)


def build_convnet_very_small(nb_filters=64, size_filters=5,
                             w=32, h=32, c=1):
    l_in = layers.InputLayer((None, w*h*c), name="input")
    x_in_reshaped = layers.ReshapeLayer(l_in, ([0], c, w, h), name="input_r")
    l_conv0 = x_in_reshaped

    nonlin = tanh 
    
    W = np.array([
        [[0, 1],
         [0, 1]],
        [[1, 1],
         [0, 0]]])
    W = W[:, None, :, :]
    W = W.astype(np.float32)
    l_conv1 = layers.Conv2DLayer(
        l_conv0,
        num_filters=nb_filters,
        filter_size=(size_filters, size_filters),
        nonlinearity=linear,
        #W=W,
        name="conv1",
    )
    l_conv1_ = layers.FeatureWTALayer(l_conv1, 2)
    #l_conv1_ = l_conv1
    l_unconv1 = layers.Conv2DLayer(
        l_conv1_,
        num_filters=1,
        filter_size=(size_filters, size_filters),
        nonlinearity=tanh,
        pad='full',
        name="unconv1",
        W=l_conv1.W.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1],
        #b=np.array([-4]).astype(np.float32)
    )

    l_out = layers.ReshapeLayer(
            l_unconv1,
            ([0], c * w * h), name="output")
    layers_ = [
        l_in,
        l_conv1,
        l_unconv1,
        l_out,
    ]
    return layers_from_list_to_dict(layers_)
