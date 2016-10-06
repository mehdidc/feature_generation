from itertools import imap
from functools import partial
import glob
from skimage.transform import resize
from skimage.util import pad
 
import numpy as np

import datakit
from datakit.helpers import dict_apply, minibatch, expand_dict, data_path

def apply_to(iterator, fn, cols=None):
    iterator = imap(partial(dict_apply, fn=fn, cols=cols), iterator)
    return iterator

def pipeline_crop(iterator, shape, pos='random', mode='constant', rng=np.random, **kw):
    crop_ = partial(crop, shape=shape, pos=pos, mode=mode, rng=rng)
    return apply_to(iterator, crop_, cols=['X'])

def crop(img, shape=(1, 1), pos='random', mode='constant', rng=np.random):
    # assumes img is shape (color, h, w)
    img_c, img_h, img_w = img.shape
    h, w = shape
    if pos == 'random':
        y = rng.randint(0, img_h)
        x = rng.randint(0, img_w)
    elif pos == 'center':
        y = img_h // 2
        x = img_w // 2
    else:
        raise Exception('Unkown mode')
    out_img = np.empty((img_c, h, w))
    img_ = np.empty((img_c, img_h + h, img_w + w))
    for c in range(img.shape[0]):
        img_[c] = pad(img[c], (h/2, w/2), str(mode))
    img = img_[:, y:y+h, x:x+w]
    return img

def pipeline_order(iterator, order='th'):
    if order == 'th':
        fn = lambda X:X.transpose((2, 0, 1))
    elif order == 'tf':
        fn = lambda X:X.transpose((1, 2, 0))
    return apply_to(iterator, fn, cols=['X'])

def pipeline_resize(iterator, shape, **kw):
    resize_ = partial(resize, output_shape=shape, preserve_range=True)
    return apply_to(iterator, resize_, cols=['X'])

def pipeline_invert(iterator, **kw):
    return apply_to(iterator, lambda x:1-x, cols=['X'])

def pipeline_normalize(iterator, value=255., **kw):
    return apply_to(iterator, lambda x:x/float(value), cols=['X'])

def pipeline_limit(iterator, nb=100):
    buffer = []
    for _ in range(nb):
        buffer.append(next(iterator))
    return buffer

def imagecollection_load_as_iterator_(pattern=''):
    pattern = data_path(pattern)
    filelist = glob.glob(pattern)
    return datakit.imagecollection.load_as_iterator(filelist)

operators = {
    'crop': pipeline_crop,
    'resize': pipeline_resize,
    'invert': pipeline_invert,
    'normalize': pipeline_normalize,
    'limit': pipeline_limit,
    'order': pipeline_order
}

datasets = {
    'mnist': datakit.mnist.load_as_iterator,
    'cifar': datakit.cifar.load_as_iterator,
    'imagecollection': imagecollection_load_as_iterator_
}

def loader(params):
    loader_name = params['name']
    loader_params = params['params']
    pipeline = params['pipeline']
    assert loader_name in datasets
    loader = datasets[loader_name]
    iterator = loader(**loader_params)
    for op in pipeline:
        name, params = op['name'], op['params']
        iterator = operators[name](iterator, **params)
    return iterator

if __name__ == '__main__':
    from tools.brushstroke.common import disp_grid
    from skimage.io import imsave
    pipeline = [
            {'name': 'limit', "params": {"nb": 10}},
            {'name': 'resize', "params": {'shape': (64, 64)}},
            {'name': 'order', "params": {"order": "th"}},
            {'name': 'crop', 'params': {'shape': (32, 32), 'pos': 'random', 'mode': 'constant'}},
            {'name': 'normalize', "params": {"value": 255}},
            {'name': 'invert', "params": {}},
    ]
    #iterator = loader({'name': 'cifar', 'params':{'which': 'train'}, 'pipeline':pipeline})
    pattern = "sketchy/256x256/sketch/tx_000000000000/**/*.png"
    iterator = loader(
        { 'name': 'imagecollection',
           'params': {'pattern': pattern},
           'pipeline': pipeline
        }
    )
    iterator = minibatch(iterator, batch_size=9)
    iterator = expand_dict(iterator)
    iterator = imap(partial(dict_apply, fn=np.array, cols=['X']), iterator)
    X = (next(iterator)['X'])
    print(X.shape)
    img = disp_grid(X, border=1, bordercolor=(0.3,0,0))
    imsave('out.png', img)
