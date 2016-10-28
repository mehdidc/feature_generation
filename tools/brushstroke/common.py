import sys
import os
sys.path.append(os.getcwd())
from tools.system.imgtovideo import imgs_to_video
from data import load_data
import numpy as np
import theano
import theano.tensor as T
from lasagne import layers as L
from tools.plot_weights import dispims_color, tile_raster_images
import pandas as pd
from tqdm import tqdm
import base64
import json
from skimage.io import imread, imsave
from skimage.transform import resize
import random
import pickle

from lightjob.cli import load_db
from lightjob.utils import dict_format as default_dict_format
from lightjob.db import SUCCESS

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

def load_model(filename, data_params=None, **kw):
    from tasks import load_
    from tasks import check as load_filename
    dataset = 'digits'
    force_w = None
    force_h = None
    params = None
    layers, params = load_(filename)
    if not data_params:
        data_params = {}
    if params:
        dataset = params.get('dataset', 'digits')
        data_params_ = params.get('data_params', {})
        data_params.update(data_params_)
        force_w = params.get('force_w')
        force_h = params.get('force_h')
    model, data, layers, w, h, c = load_filename(
            what='notebook',
            filename=filename, 
            dataset=dataset, 
            force_w=force_w, 
            force_h=force_h, 
            kw_load_data=data_params,
            **kw)
    return model, data, layers

def resize_set(x, w, h, **kw):
    x_out = np.empty((x.shape[0], 1, w, h))
    for i in range(len(x)):
        x_out[i, 0] = resize(x[i, 0], (w, h), **kw)
    return x_out.astype(np.float32)

def get_bias(layers):
    if 'biased_output' in layers:
        bias = layers['biased_output'].b.get_value()
    elif 'bias' in layers:
        bias = layers['bias'].b.get_value()
    else:
        bias = np.array(0)
        
    bias = bias[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    return bias

def get_scale(layers):
    if 'scaled_output' in layers:
        scale = layers['scaled_output'].scales.get_value()
    elif 'scale' in layers:
        scale = layers['scale'].scales.get_value()
    else:
        scale = np.array((1.,))
    scale = scale[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    return scale
    
def build_brush_func(layers, lay='brush', scale=1, bias=0, nonlin=T.nnet.sigmoid):
    X = T.tensor4()
    Bs = []
    for l in lay:
        B = L.get_output(layers[l], X)
        if len(layers[l].output_shape) == 4: # (ex, t, w, h)
            B = B.dimshuffle(0, 1, 'x', 2, 3)
        Bs.append(B)
    
    fn = theano.function(
        [X], 
        [nonlin(B * scale + bias) for B in Bs]
    )
    return fn

def build_encode_func(layers, lay='coord'):
    w = layers['output'].output_shape[2]
    X = T.tensor4()
    fn = theano.function(
        [X], 
        T.nnet.sigmoid(L.get_output(layers[lay], X)[:, :, 0:2]) * w
    )
    return fn

def build_image_to_code_func(layers, lay='coord'):
    if type(lay) != tuple:
        lay = (lay,)
    X = T.tensor4()

    outputs = [L.get_output(layers[l], X) for l in lay]

    fn = theano.function(
        [X], 
        outputs
    )
    return fn

def build_code_to_image(layers, X=T.tensor3(), lay='coord'):
    
    if type(lay) != tuple:
        lay = (lay,)
    if type(X) != tuple:
        X = (X,)
    
    inputs = {}
    
    for x, l in zip(X, lay):
        inputs[layers[l]] = x
    
    fn = theano.function(
        X, 
        L.get_output(layers['output'], inputs)
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
    from IPython.display import HTML
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

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def build_pointer_images(coord, color, w, h, p=2, sx=None, sy=None):
    color = np.array(color)
    nb_examples, T, _ = coord.shape
    imgs = np.zeros((nb_examples, T, 3, w, h))
    for e in range(nb_examples):
        for t in range(T):
            x, y = coord[e, t, :]
            x, y = int(x), int(y)
            if sx is None: w = 2
            else: w = sx[e,t]
            if sy is None: h = 2
            else: h = sy[e, t]
            p = 2
            imgs[e, t, :, y:y+1, x:x+w] = color[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            imgs[e, t, :, y+h:y+h+1, x:x+w] = color[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            imgs[e, t, :, y:y+h, x:x+1] = color[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            imgs[e, t, :, y:y+h, x+w:x+w+1] = color[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    return imgs

def build_encoders(layers, nb_parallel=None):
    lays = (['coord_{}'.format(i) for i in range(nb_parallel)] 
            if nb_parallel else ['coord'])
    lays = tuple(lays)
    encoders = []
    for lay in lays:
        encoder = build_encode_func(layers, lay=lay)
        encoders.append(encoder)
    return encoders

def find_training_job(generation_job_summary, db=None):
    if not db: db = load_db()
    s = db.get_job_by_summary(generation_job_summary)['content']['model_summary']
    return db.get_job_by_summary(s)

def find_generation_job(training_job_summary, db=None):
    if not db: db = load_db()
    jobs = db.jobs_with(state=SUCCESS, type='generation')
    for j in jobs:
        if j['content']['model_summary'] == training_job_summary:
            return j
    return None

def fast_find_generation_job(training_job_summary, db=None,cache={}):
    if not db: db = load_db()
    if not cache:
        training_jobs = db.jobs_with(state=SUCCESS, type='training')
        generation_jobs = to_generation(training_jobs, db=db)
        train_to_gen = {j_train['summary']: j_gen for j_train, j_gen in zip(training_jobs, generation_jobs)}
        cache['train_to_gen'] = train_to_gen 
    else:
        train_to_gen = cache['train_to_gen']
    return train_to_gen[training_job_summary]

def to_generation(jobs, db=None):
    if not db: db = load_db()
    S = set(j['summary'] for j in jobs)
    jobs = db.jobs_with(state=SUCCESS, type='generation')
    to_generation = {j['content']['model_summary']: j for j in jobs if j['content']['model_summary'] in S}
    jobs = map(lambda s:to_generation.get(s), S)
    return jobs

def to_training(jobs, db=None):
    if not db: db = load_db()
    return [db.get_job_by_summary(j['content']['model_summary']) for j in jobs]

def dict_format(j, field, db=None):
    if field.startswith('g#'):
        field = field[2:]
        j = fast_find_generation_job(j['summary'], db=db)
    try:
        val = default_dict_format(j, field)
        return val
    except Exception:
        return None

def preprocess_gen_data(data):
    if len(data.shape) == 5:
        data = data[:, -1] # last time step images
    if len(data.shape) == 3:
        data = data[:, np.newaxis]
    return data

def compute_sample_objectness(v):
    v = np.array(v)
    score = v * np.log(v)
    score = score.sum(axis=1)
    score = np.exp(score)
    return score

def compute_objectness(v):
    v = np.array(v)
    marginal = v.mean(axis=0)
    score = v * np.log(v / marginal)
    score = score.sum(axis=1).mean()
    score = np.exp(score)
    score = float(score)
    return score

def compute_sample_objectness_renyi(v, alpha=2):
    # source : https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy (Entropy part)
    score = (1/(alpha - 1)) * np.log((v**alpha).sum(axis=1))
    score = np.exp(score)
    return score

def compute_objectness_renyi(v, alpha=2):
    # source : https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy (divergence part)
    v = np.array(v)
    marginal = v.mean(axis=0)
    score = (1/(alpha-1)) * np.log(((v ** alpha) / marginal ** (alpha - 1)).sum(axis=1))
    score = score.mean()
    score = np.exp(score)
    score = float(score)
    return score


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def weighted_choice(objects, p, rng=random):
    cs = np.cumsum(p)
    idx = sum(cs < rng.uniform(0, 1))
    return objects[idx]

def store(x, filename):
    with open(filename, 'w') as fd:
        pickle.dump(x, fd)

def retrieve(filename):
    with open(filename) as fd:
        x = pickle.load(fd)
    return x
