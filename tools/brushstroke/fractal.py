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

import random

from cachetools import cached
load_model = cached(cache={})(load_model)

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

def gen(config):
    neuralnets = config['neuralnets']
    nb_iter = config['nb_iter']
    w = config['w']
    h = config['h']
    init = config['init']
    seed = config['seed']
    nb_snapshots = config['nb_snapshots']
    
    if nb_snapshots == 'all':
        snapshot_indices = set(range(nb_iter))
    else:
        step_size = nb_iter / nb_snapshots
        step_size = max(step_size, 1)
        snapshot_indices = range(0, nb_iter, step_size)
        if snapshot_indices[-1] != nb_iter - 1:
            snapshot_indices.append(nb_iter - 1)
        if len(snapshot_indices) > nb_iter:
            surplus = len(snapshot_indices) - nb_iter
            snapshot_indices = snapshot_indices[0:-1-surplus] + [snapshot_indices[-1]]
        print(len(snapshot_indices))
        snapshot_indices = set(snapshot_indices)

    rng = np.random.RandomState(seed)
    out_img = rng.uniform(size=(h, w)) if init == 'random' else init
    snapshots = []
    nb_full = 0
    for i in tqdm(range(nb_iter)):
        for nnet in neuralnets:
            when = nnet['when']
            if type(when) == list:
                allow = False
                for when_cur in when:
                    if type(when_cur) == tuple:
                        w1 = int(when_cur[0] * nb_iter)
                        w2 = int(when_cur[1] * nb_iter)
                        if (i >= w1 and i < w2):
                            allow = True
                    else:
                        when_cur = int(when_cur*nb_iter)
                        if i == when_cur:
                            allow = True    
                if not allow:
                    continue
            elif type(when) in (int, float):
                when = when * nb_iter
                when = int(when) + 1
                allow = (i % when) == 0
                if not allow:
                    continue
            model, _, _ = load_model(nnet['model_filename']) # done with caching so no prob, it is loaded once
            patch_h, patch_w = model.layers['output'].output_shape[2:]
            nb_iter_local = nnet['nb_iter']
            thresh = nnet['thresh']
            whitepx_ratio = nnet['whitepx_ratio']
            noise = nnet['noise']
            noise_type = nnet['noise_type']

            scale_choice = nnet['scale']
            scales = nnet['scales']
            if scale_choice == 'random':
                pr = nnet['scale_probas']
                scale_choice = rng.choice(np.arange(len(scales)), p=pr)
            else:
                pass
            scale = scales[scale_choice]

            lr_choice = nnet['learning_rate']
            if type(lr_choice) != list:
                lr_choice = [lr_choice] * len(scales)
            lr = lr_choice[scale_choice]
            
            def padwithrnd(vector, pad_width, iaxis, kw):
                return rng.uniform(size=vector.shape)
            
            def padwithzero(vector, pad_width, iaxis, kw):
                return np.zeros(vector.shape)

            img = np.lib.pad(out_img, (scale * patch_h/2, scale * patch_w/2), padwithzero)

            py_center_orig = rng.randint(0, h) # position of the center in the original out_img
            px_center_orig = rng.randint(0, w) # position of the center in the original out_img
            
            px_center = px_center_orig + patch_w / 2 # position of the center in img
            py_center = py_center_orig + patch_h / 2 # position of the center in img

            px = px_center - patch_h / 2 # position of the top-left in img
            py = py_center - patch_w / 2 # poisiton of the top-left in img

            step_y = scale
            step_x = scale

            patch = rng.uniform(size=(patch_h, patch_w))
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
            if i in snapshot_indices:
                im = out_img.copy()
                im -= im.min()
                im /= im.max()
                imsave('fractal.png', im)
                snapshots.append(out_img.copy())
    snapshots = np.array(snapshots)
    return snapshots

MODELS = {
    8:   ['b'],
    16:  ['b3', 'b2', 'a5', 'a4', 'a3', 'a2', 'a'],
    32:  ['a6'],
    64:  ['c'],
    128: ['d']
}
MODELS = {
    k: map(lambda name:'training/fractal/{}/model.pkl'.format(name), v) 
    for k, v in MODELS.items()
}

def fractal4(models=MODELS, rng=random):
    trials = []
    rng = random
    scale = 16
    nb_trials = 100000
    for i in range(nb_trials):
        model_filename = rng.choice(models[scale])
        nb_iter = 1
        when = 'always'
        w = rng.uniform(0.1, 0.5)
        thresh = rng.choice((None, 'moving'))
        learning_rate = 0.1
        seed = rng.randint(1, 1000000)
        scales = [1, 2, 3, 4]
        proba = [0.6, 0.2, 0.1, 0.1]
        neuralnets = [
            {   
                'model_filename': model_filename, 
                'nb_iter':  nb_iter, 
                'thresh': thresh, 
                'when': when, 
                'whitepx_ratio': w, 
                'scale': 'random',
                'scales': scales, 
                'scale_probas': proba,
                'learning_rate': learning_rate,
                'noise': None,
                'noise_type': None
            }
        ]
        trial_conf = {
            'neuralnets': neuralnets,
            'nb_iter': 100000,
            'w': 2**6,
            'h': 2**6,
            'init': 'random',
            'seed': seed,
            'nb_snapshots': 'all'
        }
        yield trial_conf
 

def fractal5(models=MODELS, rng=random):
    trials = []
    scale = 16
    nb_trials = 100000
    for i in range(nb_trials):
        model_filename = rng.choice(models[scale])
        scales = [1, 2]
        pr = rng.uniform(0, 1)
        proba = [pr, 1 - pr]
        neuralnets = [
            {   
                'model_filename': model_filename, 
                'nb_iter':  1, 
                'thresh': rng.choice((None, 'moving')), 
                'when': 'always', 
                'whitepx_ratio': rng.uniform(0.1, 0.5), 
                'scale': 'random',
                'scales': scales, 
                'scale_probas': proba,
                'learning_rate': 0.3,
                'noise': None,
                'noise_type': None
            }
        ]
        trial_conf = {
            'neuralnets': neuralnets,
            'nb_iter': 20000,
            'w': 2**8,
            'h': 2**8,
            'init': 'random',
            'seed': rng.randint(1, 10000000),
            'nb_snapshots': 1000
        }
        yield trial_conf

def fractal6(models=MODELS, rng=random):
    trials = []
    scale = 16
    nb_trials = 100000
    for i in range(nb_trials):
        model_filename = rng.choice(models[scale])
        scales = [1, 2, 3]
        pr1 = rng.uniform(0, 1)
        pr2 = rng.uniform(0, 1)
        proba = [pr1, pr2, 1 - pr1 - pr2]
        neuralnets = [
            {   
                'model_filename': model_filename, 
                'nb_iter':  1, 
                'thresh': rng.choice((None, 'moving')), 
                'when': 'always', 
                'whitepx_ratio': rng.uniform(0.1, 0.5), 
                'scale': 'random',
                'scales': scales, 
                'scale_probas': proba,
                'learning_rate': 0.3,
                'noise': None,
                'noise_type': None
            }
        ]
        trial_conf = {
            'neuralnets': neuralnets,
            'nb_iter': 20000,
            'w': 2**8,
            'h': 2**8,
            'init': 'random',
            'seed': rng.randint(1, 10000000),
            'nb_snapshots': 1000
        }
        yield trial_conf
 
def silent_create_folder(folder):
    try:
        os.mkdir(folder)
    except OSError:
        pass

if __name__ == '__main__':
    # center
    # black/white inversion
    # video
    from docopt import docopt
    import random
    import os
    import json

    doc = """
    Usage: fractal.py JOB [FOLDER]
    
    MODE: serial
    """
    args = docopt(doc)
    job = args['JOB']
    serial_sample = globals()[job]
    folder = args['FOLDER']
    if not folder: folder = 'exported_data/fractal/{}'.format(job)
    print(folder)
    silent_create_folder(folder)
    silent_create_folder(folder + '/videos')
    silent_create_folder(folder + '/images')
    random.seed(42)
    trials = []
    i = 0
    for trial_conf in serial_sample(models=MODELS):
        trials.append(trial_conf)
        trial_conf['folder'] = folder
        with open('{}/trials.json'.format(folder), 'w') as fd:
            fd.write(json.dumps(trials, indent=2))
        snaps = gen(trial_conf)
        snaps -= snaps.min(axis=(1,2), keepdims=True)
        snaps /= snaps.max(axis=(1,2), keepdims=True)
        image_filename = '{}/images/trial{:05d}.png'.format(folder, i)
        img = snaps[-1]
        imsave(image_filename, img)
        snaps = snaps[None, :, None, :, :]
        video_filename = '{}/videos/trials{:05d}.mp4'.format(folder, i)
        seq_to_video(snaps, filename=video_filename, framerate=10, rate=10)
        i += 1
