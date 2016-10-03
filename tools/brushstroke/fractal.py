from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.io import imsave
from skimage.util import pad
from common import load_model, disp_grid
import sys
import os
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

def gen(neuralnets, nb_iter=10, w=32, h=32, init='random', out='out.png', rng=np.random):
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
            img = out_img
            noise = nnet.get('noise', 0)
            noise_type = nnet.get('noise_type', 'zero_masking')
            scale = nnet.get('scale', 1)
            if scale == 'random':
                scale = rng.randint(*nnet.get('scale_range', 4))
            py = rng.randint(0, img.shape[0])
            px = rng.randint(0, img.shape[1])
            
            step_y = scale
            step_x = scale
            patch = np.random.uniform(size=(patch_h, patch_w))
            crop = img[py:py + patch_h*step_y:step_y, px:px + patch_w*step_x:step_x]
            patch[:crop.shape[0], :crop.shape[1]] = crop
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
            shape = img[py:py + p.shape[0] * step_y:step_y, px:px + p.shape[1] * step_x:step_x].shape
            p_cropped = p[:shape[0], :shape[1]]
            out_img[py:py + p.shape[0] * step_y:step_y, px:px + p.shape[1] * step_x:step_x] = p_cropped
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

    for s in scale_128_128:
        model, data, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[128].append((s, model))


    for s in scale_64_64:
        model, data, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[64].append((s, model))

    for s in scale_32_32:
        model, data, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[32].append((s, model))

    for s in scale_16_16:
        model, data, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[16].append((s, model))

    for s in scale_8_8:
        model, data, layers = load_model('training/fractal/{}/model.pkl'.format(s))
        models[8].append((s, model))
    nb_trials = 100000
    import random
    trials = []
    rng = random
    for i in range(nb_trials):
        sa, model_a  = rng.choice(models[8])
        sb, model_b  = rng.choice(models[16])
        sc, model_c = rng.choice(models[32])
        sd, model_d = rng.choice(models[64])
        se, model_e = rng.choice(models[128])

        nb_iter_a = rng.randint(1, 20)
        nb_iter_b = rng.randint(1, 20)
        nb_iter_c = rng.randint(1, 20)
        nb_iter_d = rng.randint(1, 20)
        nb_iter_e = rng.randint(1, 20)


        whena = rng.choice(  ('always',  rng.uniform(0, 0.5)   )   )
        whenb = rng.choice(  ('always',  rng.uniform(0, 0.5)   )   )
        whenc = rng.choice(  ('always',  rng.uniform(0, 0.5)   )   )
        whend = rng.choice(  ('always',  rng.uniform(0, 0.5)   )   )
        whene = rng.choice(  ('always',  rng.uniform(0, 0.5)   )   )

        wa = rng.uniform(0.1, 0.5)
        wb = rng.uniform(0.1, 0.5)
        wc = rng.uniform(0.1, 0.5)
        wd = rng.uniform(0.1, 0.5)
        we = rng.uniform(0.1, 0.2)

        trial_conf = [sa, sb, sc, sd, nb_iter_a, nb_iter_b, nb_iter_c, nb_iter_d, whena, whenb, whenc, whend, whene, wa, wb, wc, wd, we]
        trials.append(trial_conf)
        with open('exported_data/fractal/trials.json', 'w') as fd:
            fd.write(json.dumps(trials))
        neuralnets = [
            {'model': model_a, 'on': 'crops', 'padlen': 3,   'nb_iter':  nb_iter_a,   'thresh': 'moving', 'when': whena, 'whitepx_ratio': wa},
            {'model': model_b, 'on': 'crops', 'padlen': 3,   'nb_iter':  nb_iter_b,   'thresh': 'moving', 'when': whenb, 'whitepx_ratio': wb},
            {'model': model_c, 'on': 'crops', 'padlen': 3,   'nb_iter':  nb_iter_c,   'thresh': 'moving', 'when': whenc, 'whitepx_ratio': wc},
            {'model': model_d, 'on': 'crops', 'padlen': 3,   'nb_iter':  nb_iter_d,   'thresh': 'moving', 'when': whend, 'whitepx_ratio': wd},
            {'model': model_e, 'on': 'crops', 'padlen': 3,   'nb_iter':  nb_iter_e,   'thresh': 'moving', 'when': whene, 'whitepx_ratio': we},
        ]
        img, snap = gen(neuralnets, nb_iter=5000, w=2**7, h=2**7, init='random')
        imsave('exported_data/fractal/trial{:05d}.png'.format(i), img)
     
if __name__ == '__main__':
    from docopt import docopt
    doc = """
    Usage: fractal.py MODE

    Arguments:
    MODE serial/manual
    """
    args = docopt(doc)
    mode = args['MODE']
    if mode == 'serial':
        serialrun()
    elif mode == 'manual':
        model_a, data, layers = load_model("training/fractal/a6/model.pkl")
        neuralnets = [
                { 'model': model_a, 
                  'nb_iter':  5,
                  'thresh': 'moving', 
                  'when': 'always', 
                  'whitepx_ratio': 0.2, 
                  'scale': 'random', 
                  'scale_range':(1, 10)},
        ]
        imgs = []
        img, snap = gen(neuralnets, nb_iter=1000, w=2**5, h=2**5, init='random', out='manual.png')
        imgs = img[np.newaxis, np.newaxis, :, :]
        img = disp_grid(imgs, border=1, bordercolor=(0.3, 0, 0))
        imsave('grid.png', img)
    else:
        raise Exception('Unknown mode : {}'.format(mode))
