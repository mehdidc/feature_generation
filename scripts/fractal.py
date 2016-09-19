import sys, os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = 'device=cpu,compiledir_format="ipynb_compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s"'
sys.path.append("/home/mcherti/work/code/feature_generation")
from tasks import check as load_filename
from scripts.imgtovideo import imgs_to_video
from data import load_data
from helpers import salt_and_pepper
import matplotlib.pyplot as plt
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
from skimage.util import pad
from tqdm import tqdm

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
def normalize(img):
    img = img.copy()
    img -= img.min()
    img /= img.max()
    return img

def downscale(x, scale=2):
    h, w = x.shape
    x = x.reshape((h / scale, scale, w / scale, scale))
    return x.mean(axis=(1, 3)), (x - x.mean(axis=(1, 3), keepdims=True)).reshape((h, w))

def downscale_simple(x, scale=2):
    h, w = x.shape
    return x[::scale, ::scale]    
    
def upscale(x, r):
    shape = r.shape
    y_scale = r.shape[0] / x.shape[0]
    x_scale = r.shape[1] / x.shape[1]        
    r = r.reshape((r.shape[0] / y_scale, y_scale, r.shape[1] / x_scale, x_scale))
    x = x.reshape((x.shape[0], 1, x.shape[1], 1))
    return (x + r).reshape(shape)

def upscale_simple(x, scale=2):
    y = np.zeros((x.shape[0]*scale, x.shape[1]*scale))
    y[::scale, ::scale] = x
    return y

def gen(neuralnets, nb_iter=10, w=32, h=32, init='random'):
    out_img = np.random.uniform(size=(h, w)) if init == 'random' else init
    snapshots = []
    nb_full = 0
    for i in tqdm(range(nb_iter)):
        #snapshots.append(out_img.copy())
        for nnet in neuralnets:
            when = nnet.get('when', 'always')
            if type(when) == list:
                allow = False
                for w in when:
                    if type(w) == tuple:
                        w1 = int(w[0] * nb_iter)
                        w2 = int(w[1] * nb_iter)
                        if (i >= w1 and i < w2):
                            allow = True
                    else:
                        w = int(w*nb_iter)
                        if i == w:
                            allow = True    
                if not allow:
                    continue
            model = nnet['model']
            on = nnet['on']
            patch_h, patch_w = model.layers['output'].output_shape[2:]
            nb_iter_local = nnet.get('nb_iter', 10)
            thresh = nnet.get('thresh', 0.5)
            whitepx_ratio = nnet.get('whitepx_ratio', 0.5)
            if on == 'crops':
                padlen = nnet.get('padlen', 5) 
                img = pad(out_img, padlen, 'constant', constant_values=(0, 0))
                py = np.random.randint(0, img.shape[0] - patch_h)
                px = np.random.randint(0, img.shape[1] - patch_w)
                patch = img[py:py + patch_h, px:px + patch_w]
                patch = patch[np.newaxis, np.newaxis, :, :]
                patch = patch.astype(np.float32)
                noise = nnet.get('noise', 0)
                for _ in range(nb_iter_local):
                    if noise:patch *= np.random.uniform(size=patch.shape)<=(1-noise)
                    patch = model.reconstruct(patch)
                    
                    if thresh == 'moving':
                        vals = patch.flatten()
                        vals = vals[np.argsort(vals)]
                        thresh_ = vals[-int(whitepx_ratio * len(vals)) - 1]
                    else:
                        thresh_ = thresh
                    if thresh: patch = patch > thresh_
                    patch = patch.astype(np.float32)
                img[py:py + patch_h, px:px + patch_w] = patch[0, 0]
                out_img[:, :] = img[padlen:-padlen, padlen:-padlen]
            elif on == 'full':
                nb_full += 1
                scale = h / model.layers['output'].output_shape[2]
                #img, resid = downscale(out_img, scale=scale)
                img = downscale_simple(out_img, scale=scale)
                img = img[np.newaxis, np.newaxis, :, :]
                img = img.astype(np.float32)
                for _ in range(nb_iter_local):
                    img = model.reconstruct(img)
                    
                    if thresh == 'moving':
                        vals = img.flatten()
                        vals = vals[np.argsort(vals)]
                        thresh_ = vals[-int(whitepx_ratio * len(vals)) - 1]
                    else:
                        thresh_ = thresh

                    if thresh:img = img > thresh_
                    
                img = img[0, 0]
                #resid[:]=0
                #print(img.shape)
                img = upscale_simple(img, scale=scale)
                #print(img.shape)

                #img = normalize(img)
                img = img > thresh_
                img = img.astype(np.float32)
                out_img = img
            if i % 100 == 0:
                imsave('out.png', img)
    return out_img, snapshots

if __name__ == '__main__':
    model_a, data, layers, w, h, c = load_model("training/fractal/a/model.pkl", 
                                                dataset="rescaled_digits", 
                                                force_w=16, force_h=16)

    model_b, data, layers, w, h, c = load_model("training/fractal/b/model.pkl", 
                                                dataset="random_cropped_digits", 
                                                force_w=8, force_h=8)
    neuralnets = [
        {'model': model_b, 'on': 'crops', 'padlen': 3,   'nb_iter':  5,   'thresh': 0.5, 'when': 'always', 'whitepx_ratio': 0.5},
        #{'model': model_a, 'on': 'crops',  'padlen': 3,  'nb_iter': 5,   'thresh': 'moving', 'when': 'always', 'whitepx_ratio': 0.185},
    ]
    imgs = []
    for i in range(1):
        img, snap = gen(neuralnets, nb_iter=20000, w=2**9, h=2**9, init='random')
        imgs.append(img)
