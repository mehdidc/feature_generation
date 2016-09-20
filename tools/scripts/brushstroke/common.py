import sys
import os
sys.path.append(os.getcwd())
from tasks import load_
from tasks import check as load_filename
from scripts.imgtovideo import imgs_to_video
from data import load_data
import numpy as np
import theano
import theano.tensor as T
from lasagne import layers as L
from lasagnekit.misc.plot_weights import dispims_color, tile_raster_images
import pandas as pd
from tqdm import tqdm
import base64
import json
from skimage.io import imread, imsave
from skimage.transform import resize

def minibatcher(fn, batchsize=1000):
  """
  fn : a function that takes an input and returns an output
  batchsize : divide the total input into divisions of size batchsize at most
  
  iterate through all the divisions, call fn, get the results, 
  then concatenate all the results.
  """
  def f(X):
      results = []
      for sl in iterate_minibatches(len(X), batchsize):
          results.append(fn(X[sl]))
      return np.concatenate(results, axis=0)
  return f

def iterate_minibatches(nb_inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(nb_inputs)
        np.random.shuffle(indices)
    for start_idx in range(0, max(nb_inputs, nb_inputs - batchsize + 1), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield excerpt

def load_model(filename, **kw):
    dataset = 'digits'
    force_w = None
    force_h = None
    params = None
    layers, params = load_(filename)
    if params:
        dataset = params.get('dataset', 'digits')
        force_w = params.get('force_w')
        force_h = params.get('force_h')
    model, data, layers, w, h, c = load_filename(
            what='notebook',
            filename=filename, 
            dataset=dataset, 
            force_w=force_w, 
            force_h=force_h, 
            kw_load_data={'include_test': True},
            **kw)
    return model, data, layers

def build_brush_func(layers):
    if 'biased_output' in layers:
        bias = layers['biased_output'].b.get_value()
    elif 'bias' in layers:
        bias = layers['bias'].b.get_value()
    else:
        bias = np.array(0.1)

    bias = bias[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    if 'scaled_output' in layers:
        scale = layers['scaled_output'].scales.get_value()
    elif 'scale' in layers:
        scale = layers['scale'].scales.get_value()
    else:
        scale = np.array((1.,))
    scale = scale[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    
    X = T.tensor4()

    B = L.get_output(layers['brush'], X)
    if len(layers['brush'].output_shape) == 4: # (ex, t, w, h)
        B = B.dimshuffle(0, 1, 'x', 2, 3)
    
    fn = theano.function(
        [X], 
        T.nnet.sigmoid(B * scale + bias)
    )
    return fn

def build_encode_func(layers):
    w = layers['output'].output_shape[2]
    X = T.tensor4()
    fn = theano.function(
        [X], 
        T.nnet.sigmoid(L.get_output(layers['coord'], X)[:, :, 0:2]) * w
    )
    return fn

def to_grid_of_images(seq_imgs, **kw):
    y = seq_imgs
    imgs = []
    for t in range(y.shape[1]):
        yy = y[:, t]
        if yy.shape[1] == 1:
            yy = yy[:, 0, :, :, np.newaxis] * np.ones((1, 1, 1, 3))
        else:
            yy = yy.transpose((0, 2, 3, 1))
        img = dispims_color(yy, **kw)
        imgs.append(img)
    return imgs

def seq_to_video(seq, filename='out.mp4', verbose=1, framerate=8, rate=8, **kw):
    # shape of seq should be : (examples, time, c, w, h)
    seq = to_grid_of_images(seq, **kw)
    seq = [np.zeros_like(seq[0])] + seq
    if os.path.exists(filename):
        os.remove(filename)
    imgs_to_video(seq, out=filename, verbose=verbose, framerate=framerate, rate=rate)

def embed_video(filename):
    video = open(filename, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii')))

def disp_grid(imgs, **kw):
    # shape of imgs should be : (examples, color, w, h)
    out = dispims_color(imgs.transpose((0, 2, 3, 1)) * np.ones((1, 1, 1, 3)), **kw)
    return out


def prop_uniques(x):
    x = x.reshape((x.shape[0], -1))
    x = map(hash_array, x)
    return len(set(x)) / float(len(x))

def hash_array(x):
    return hash(tuple(x))

def normalize(x, axis=1):
    return (x - x.min(axis=axis, keepdims=True)) / (x.max(axis=axis, keepdims=True) - x.min(axis=axis, keepdims=True))
