from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.io import imsave
from skimage.util import pad
from common import load_model, disp_grid

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

def gen(neuralnets, nb_iter=10, w=32, h=32, init='random', out='out.png'):
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
                inner_padlen = nnet.get('inner_padlen', 0)
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
                #if np.random.uniform() <= 0.2:
                #    patch = 1 - patch
                if inner_padlen:
                    p = patch[:, :, inner_padlen:-inner_padlen, inner_padlen:-inner_padlen]
                    #print(p.shape, patch_h - inner_padlen * 2)
                else:
                    p = patch
                img[py:py + patch_h - inner_padlen*2, px:px + patch_w - inner_padlen*2] = p 
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
                imsave(out, img)
    return out_img, snapshots

def serialrun():
    from collections import defaultdict
    import json

    scale_32_32 = ['a6']
    scale_16_16 = ['b3', 'b2', 'a5', 'a4', 'a3', 'a2', 'a']
    scale_8_8 = ['b']
    
    models = defaultdict(list)

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
        nb_iter_a = rng.randint(1, 20)
        nb_iter_b = rng.randint(1, 20)
        nb_iter_c = rng.randint(1, 20)
        whena = rng.choice(('always', [0], [0, 0.5]))
        whenb = rng.choice(('always', [0], [0, 0.5]))
        whenc = rng.choice(('always', [0], [0, 0.5]))
        wa = rng.uniform(0.1, 0.5)
        wb = rng.uniform(0.1, 0.5)
        wc = rng.uniform(0.1, 0.5)
        trial_conf = [sa, sb, sc, nb_iter_a, nb_iter_b, nb_iter_c, whena, whenb, whenc]
        trials.append(trial_conf)
        with open('exported_data/fractal/trials.json', 'w') as fd:
            fd.write(json.dumps(trials))
        neuralnets = [
            {'model': model_a, 'on': 'crops', 'padlen': 3,   'nb_iter':  nb_iter_a,   'thresh': 'moving', 'when': whena, 'whitepx_ratio': wa},
            {'model': model_b, 'on': 'crops', 'padlen': 3,   'nb_iter':  nb_iter_b,   'thresh': 'moving', 'when': whenb, 'whitepx_ratio': wb},
            {'model': model_c, 'on': 'crops', 'padlen': 3,   'nb_iter':  nb_iter_c,   'thresh': 'moving', 'when': whenc, 'whitepx_ratio': wc},
        ]
        img, snap = gen(neuralnets, nb_iter=2000, w=2**7, h=2**7, init='random')
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
        model_a, data, layers = load_model("training/fractal/b/model.pkl")
        model_b, data, layers = load_model("training/fractal/b2/model.pkl")
        model_c, data, layers = load_model("training/initial_models/model_E.pkl")
        neuralnets = [
            {'model': model_a, 'on': 'crops', 'padlen': 3,   'nb_iter':  5,   'thresh': 'moving', 'when': 'always', 'whitepx_ratio': 0.1},
            {'model': model_b, 'on': 'crops', 'padlen': 3,   'nb_iter':  5,   'thresh': 'moving', 'when': 'always', 'whitepx_ratio': 0.2},
            {'model': model_c, 'on': 'crops', 'padlen': 3,   'nb_iter':  5,   'thresh': 'moving', 'when': 'always', 'whitepx_ratio': 0.1},
        ]
        imgs = []
        for i in range(1):
            img, snap = gen(neuralnets, nb_iter=2000, w=2**7, h=2**7, init='random', out='manual.png')
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs[:, np.newaxis, :, :]
        print(imgs.shape)
        img = disp_grid(imgs, border=1, bordercolor=(0.3, 0, 0))
        imsave('grid.png', img)
    else:
        raise Exception('Unknown mode : {}'.format(mode))
