import sys, os
from docopt import  docopt
sys.path.append("/home/mcherti/work/code/feature_generation")
from tasks import load_
from tasks import check as load_filename
from scripts.imgtovideo import imgs_to_video
from data import load_data

import numpy as np

import theano
import theano.tensor as T

from lasagne import layers as L

from lasagnekit.misc.plot_weights import dispims_color, tile_raster_images

from IPython.display import HTML, Image

import pandas as pd

from tqdm import tqdm

import base64
import json

from skimage.io import imread, imsave
from skimage.transform import resize

def load_model(filename, **kw):

    model = load_filename(
        what="notebook", 
        filename=filename, 
        **kw
    )
    return model

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



def main():
    doc = """
    Usage:
        export_img.py [--from-json=JSON] [--id=ID] FILE [OUTPUT]

    Arguments:
        FILE input file
        OUTPUT output directory

    Options:
        -h --help     Show this screen
        --from-json=JSON json configuration file
        --id=ID id of job
    """

    args = docopt(doc)
    dataset = 'digits'
    force_w = None
    force_h = None
    params = None
    
    layers, params = load_(args['FILE'])#if w not specified take the one in the model
    if params:
        dataset = params.get('dataset', 'digits')
        force_w = params.get('force_w')
        force_h = params.get('force_h')
    model, data, layers, w, h, c = load_model(args['FILE'], dataset=dataset, force_w=force_w, force_h=force_h, kw_load_data={'include_test': True})
    
    O = args['OUTPUT'] + '/'
    encode = build_encode_func(layers) # transforms image to sequence of coordinates
    brush = build_brush_func(layers) # transforms an image to sequence of images
    X = data.X[0:11*11]
    imgs = brush(model.preprocess(X)) # (examples, time, w, h)
    seq_to_video(imgs, O+'seq.mp4')

    im1 = disp_grid(model.preprocess(data.X[0:100]), border=1, bordercolor=(.3, .3, .3))
    im2 = disp_grid(model.reconstruct(model.preprocess(data.X[0:100])), border=1, bordercolor=(.5, 0, 0))
    im_mix = np.empty((im1.shape[0], im1.shape[1] + im2.shape[1], 3))
    im_mix[:, 0:im1.shape[1]] = im1
    im_mix[:, im1.shape[1]:] = im2
    imsave(O + 'im_mix.png', im_mix)

    np.random.seed(2)
    
    nb_iter = 10
    nb_examples = 100
    thresh = 0.3
    use_noise = False

    # PREP
    if use_noise: noise = np.random.normal(0, 0.5, size=imgs[:, 0].shape).astype(np.float32) #(for colored images)
    else: noise = 0
    if thresh == 'moving':
        whitepx_ratio = (data.X>0.5).sum() / np.ones_like(data.X).sum()
    imgs = np.empty((nb_examples, nb_iter + 1, c, w, h)) # 1 = color channel
    imgs = imgs.astype(np.float32)
    #imgs[:, 0] = data.X[0:nb_examples].reshape((nb_examples, 1, h, w)) >0.5
    imgs[:, 0] = np.random.uniform(size=(nb_examples, c, w, h))

    scores = []
    diversities = []

    # ITERATIOn

    for i in tqdm(range(1, nb_iter + 1)):
        
        if use_noise:noise = np.random.normal(0, 1, size=imgs[:, 0].shape).astype(np.float32) #(for colored images)
        else:noise = 0
            
        imgs[:, i] = brush(imgs[:, i - 1] + noise)[:,-1]
        if c == 1:
            if thresh == 'moving':
                vals = imgs[:, i].flatten()
                vals = vals[np.argsort(vals)]
                thresh_ = vals[-int(whitepx_ratio * len(vals)) - 1]
            else:
                thresh_ = thresh
            if thresh_:
                imgs[:, i] = imgs[:, i] > thresh_ # binarize
        score = np.abs(imgs[:, i - 1] - imgs[:, i]).sum()
        scores.append(score)
        diversity = prop_uniques(imgs[:, i])
        diversities.append(diversity)
    
    img = disp_grid(imgs[:, -1], border=1, bordercolor=(0.3, 0, 0))
    imsave(O + 'ir.png', img)

    seq_to_video(imgs, O+'ir.mp4', border=0, bordercolor=(0, 0, 0))

    dt_test = load_data('omniglot', w=w, h=h)
    # load from file

    nb = 100
    dt = dt_test.X[0:nb]
    try:
        dt = dt.reshape((nb, c, w, h))
    except Exception:
        dt = dt.reshape((nb, 1, w, h))
        dt = dt * np.ones((1, 3, 1, 1))
        dt = dt.astype(np.float32)
    print(dt.shape)
    rec = model.reconstruct(dt)
    print(((rec - dt)**2).mean())
    im1 = disp_grid(model.preprocess(dt[0:nb]), border=1, bordercolor=(.3, .3, .3))
    im2 = disp_grid(model.reconstruct(model.preprocess(rec[0:nb])), border=1, bordercolor=(.5, 0, 0))
    im_mix = np.empty((im1.shape[0], im1.shape[1] + im2.shape[1], 3))
    im_mix[:, 0:im1.shape[1]] = im1
    im_mix[:, im1.shape[1]:] = im2
    imsave(O+'im_mix_new_dataset.png', im_mix)

if __name__ == '__main__':
    main()
