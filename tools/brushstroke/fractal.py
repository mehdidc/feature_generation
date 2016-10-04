from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.io import imsave
from skimage.util import pad
from common import load_model, disp_grid, seq_to_video
import sys
import os
import shutil

sys.path.append(os.path.dirname(__file__)  + '/../../')
from helpers import salt_and_pepper
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

def gen(neuralnets, nb_iter=10, w=32, h=32, init='random', out='out.png', rng=np.random, video=True):
    out_img = rng.uniform(size=(h, w)) if init == 'random' else init
    snapshots = []
    nb_full = 0
    for i in tqdm(range(nb_iter)):
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
            elif type(when) in (int, float):
                when = when * nb_iter
                when = int(when) + 1
                allow = (i % when) == 0
                if not allow:
                    continue
            model = nnet['model']
            patch_h, patch_w = model.layers['output'].output_shape[2:]
            nb_iter_local = nnet.get('nb_iter', 10)
            thresh = nnet.get('thresh', 0.5)
            whitepx_ratio = nnet.get('whitepx_ratio', 0.5)
            lr = nnet.get('learning_rate', 0.1)

            noise = nnet.get('noise', 0)
            noise_type = nnet.get('noise_type', 'zero_masking')
            scale = nnet.get('scale', 1)
            if scale == 'random':
                pr = nnet.get('scale_probas', [1])
                scales = nnet.get('scales', [1])
                scale = rng.choice(scales, p=pr)
            
            def padwithrnd(vector, pad_width, iaxis, kw):
                return np.random.uniform(size=vector.shape)

            img = np.lib.pad(out_img, (scale * patch_h/2, scale * patch_w/2), padwithrnd)

            py_center_orig = rng.randint(0, h) # position of the center in the original out_img
            px_center_orig = rng.randint(0, w) # position of the center in the original out_img
            
            px_center = px_center_orig + patch_w / 2 # position of the center in img
            py_center = py_center_orig + patch_h / 2 # position of the center in img

            px = px_center - patch_h / 2 # position of the top-left in img
            py = py_center - patch_w / 2 # poisiton of the top-left in img

            step_y = scale
            step_x = scale

            patch = np.random.uniform(size=(patch_h, patch_w))
            crop = img[py:py + patch_h*step_y:step_y, px:px + patch_w*step_x:step_x]
            patch[:, :] = crop
            patch = patch[np.newaxis, np.newaxis, :, :]
            patch = patch.astype(np.float32)
            for _ in range(nb_iter_local):
                if noise:
                    if noise_type == 'zero_masking': patch *= rng.uniform(size=patch.shape)<=(1-noise)
                    if noise_type == 'salt_and_pepper': patch = salt_and_pepper(patch, backend='numpy', corruption_level=noise)
                patch = model.reconstruct(patch)
                if thresh == 'moving':
                    vals = patch.flatten()
                    vals = vals[np.argsort(vals)]
                    thresh_ = vals[-int(whitepx_ratio * len(vals)) - 1]
                else:
                    thresh_ = thresh
                if thresh: patch = patch > thresh_
                patch = patch.astype(np.float32)
            p = patch[0, 0]
            prev = img[py:py + patch_h*step_y:step_y, px:px + patch_w*step_x:step_x]
            new = p
            img[py:py + patch_h*step_y:step_y, px:px + patch_w*step_x:step_x] = (prev * (1 - lr) + new * lr)
            out_img[:] = img[scale*patch_h/2:-scale*patch_h/2, scale*patch_w/2:-scale*patch_w/2]
            snapshots.append(out_img.copy())
            if i % 1000==0 and video:
                s = np.array(snapshots)
                s = s[None, :, None, :, :]
                seq_to_video(s, filename='out.mp4')
                shutil.copy('out.mp4', 'cur.mp4')
    return out_img, snapshots

def serialrun():
    from collections import defaultdict
    import json
    scale_128_128 = ['d']
    scale_64_64 = ['c']
    scale_32_32 = ['a6']
    scale_16_16 = ['b3', 'b2', 'a5', 'a4', 'a3', 'a2', 'a']
    scale_8_8 = ['b']
    models = defaultdict(list)
    """
    for s in scale_128_128:
        model, data, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[128].append((s, model))

    for s in scale_64_64:
        model, data, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[64].append((s, model))

    for s in scale_32_32:
        model, d/ata, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[32].append((s, model))
    """
    for s in scale_16_16:
        model, data, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[16].append((s, model))
    """
    for s in scale_8_8:
        model, data, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[8].append((s, model))
    """
    nb_trials = 100000
    import random
    trials = []
    rng = random
    for i in range(nb_trials):
        scale = 16
        name, model = rng.choice(models[scale])
        nb_iter = 1
        when = 'always'
        w = rng.uniform(0.1, 0.5)
        thresh = rng.choice((None, 'moving'))
        learning_rate = 0.1

        trial_conf = {
            'name': name, 
            'nb_iter': nb_iter, 
            'when': when, 
            'whitepx_ratio': w, 
            'thresh': thresh, 
            'learning_rate': learning_rate
        }
        trials.append(trial_conf)
        with open('exported_data/fractal/trials.json', 'w') as fd:
            fd.write(json.dumps(trials, indent=4))
        proba = [0.6, 0.2, 0.1, 0.1]
        scales = [1, 2, 3, 4]
        neuralnets = [
            {'model': model, 'nb_iter':  nb_iter, 'thresh': thresh, 'when': when, 'whitepx_ratio': w, 'scales': scales, 'scale_probas': proba},
        ]
        img, snap = gen(neuralnets, nb_iter=10000, w=2**6, h=2**6, init='random', video=False)
        img -= img.min()
        img /= img.max()
        imsave('exported_data/fractal/trial{:05d}.png'.format(i), img)
     
if __name__ == '__main__':
    # center
    # black/white inversion
    # video
    from docopt import docopt
    doc = """
    Usage: fractal.py MODE

    Arguments:
    MODE serial/manual
    """
    args = docopt(doc)
    mode = args['MODE']
    if mode == 'serial':
        np.random.seed(42)
        serialrun()
    elif mode == 'manual':
        model_a, data, layers = load_model("training/fractal/a6/model.pkl")
        neuralnets = [
                { 'model': model_a, 
                  'nb_iter':  1,
                  'thresh': 'moving', 
                  'when': 'always', 
                  'whitepx_ratio': 0.2, 
                  'scale': 'random', 
                  'scale_range':(1,5)},
        ]
        imgs = []
        img, snap = gen(neuralnets, nb_iter=100000, w=2**6, h=2**6, init='random', out='manual.png')
        imgs = img[np.newaxis, np.newaxis, :, :]
        img = disp_grid(imgs, border=1, bordercolor=(0.3, 0, 0))
        imsave('grid.png', img)
    else:
        raise Exception('Unknown mode : {}'.format(mode))
