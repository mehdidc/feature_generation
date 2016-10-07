from itertools import imap
from functools import partial
import random
import glob

from skimage.transform import resize
from skimage.util import pad
from skimage.io import imread

import numpy as np

import datakit
from datakit.helpers import dict_apply, minibatch, expand_dict, data_path

dataset_patterns = {
    'sketchy': 'sketchy/256x256/sketch/tx_000000000000/**/*.png',
    'flaticon': 'flaticon/**/png/*.png',
    'lfw': 'lfw/imgaligned/**/*.jpg',
    'shoes': 'shoes/ut-zap50k-images/Shoes/**/**/*.jpg',
    'svhn': 'svhn/**/*.png',
    'chairs': 'chairs/rendered_chairs/**/renders/*.png',
    'icons': 'icons/img/*.png',
    'aloi': 'aloi/png4/**/*.png',
    'kanji': 'kanji/cleanpngsmall/*.png',
    'iam': 'iam/**/**/*.png',
}


def apply_to(iterator, fn, cols=None):
    iterator = imap(partial(dict_apply, fn=fn, cols=cols), iterator)
    return iterator

def as_iterator_func(fn):
    def fn_(iterator, *args, **kwargs):
        return apply_to(iterator, partial(fn, *args, **kwargs), cols=['X'])
    return fn_

def crop(img, shape=(1, 1), pos='random', mode='constant', rng=np.random):
    # assumes img is shape (h, w, color)
    assert len(img.shape) == 3 and img.shape[2] in (1, 3)
    img_h, img_w, img_c = img.shape
    h, w = shape
    if pos == 'random':
        y = rng.randint(0, img_h)
        x = rng.randint(0, img_w)
    elif pos == 'center':
        y = img_h // 2
        x = img_w // 2
    else:
        raise Exception('Unkown mode')
    out_img = np.empty((h, w, img_c))
    img_ = np.empty((img_h + h, img_w + w, img_c))
    for c in range(img_c):
        img_[:, :, c] = pad(img[:, :, c], (h/2, w/2), str(mode))
    img = img_[y:y+h, x:x+w, :]
    return img

def pipeline_crop(iterator, shape, pos='random', mode='constant', rng=np.random, **kw):
    crop_ = partial(crop, shape=shape, pos=pos, mode=mode, rng=rng)
    return apply_to(iterator, crop_, cols=['X'])

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

def pipeline_divide_by(iterator, value=255., **kw):
    return apply_to(iterator, lambda x:x/float(value), cols=['X'])

def pipeline_limit(iterator, nb=100):
    buffer = []
    for _ in range(nb):
        buffer.append(next(iterator))
    return buffer

def pipeline_shuffle(iterator, rng=random):
    iterator = list(iterator)
    rng.shuffle(iterator)
    return iter(iterator)

def pipeline_imagefilelist(iterator, pattern=''):
    pattern = pattern.format(**dataset_patterns)
    pattern = data_path(pattern)
    filelist = glob.glob(pattern)
    return iter(filelist)

def pipeline_imageread(iterator):
    return datakit.imagecollection.load_as_iterator(iterator)

def pipeline_normalize_shape(iterator):
    # if shape = 2, add a new axis at the right
    # if shape = 3, leave it as it is
    return apply_to(iterator, lambda x:x[:, :, np.newaxis] if len(x.shape) == 2 else x, cols=['X'])

def pipeline_load_dataset(iterator, name, *args, **kwargs):
    assert hasattr(datakit, name)
    module = getattr(datakit, name)
    return module.as_iterator(*args, **kwargs)

operators = {
    'dataset': pipeline_load_dataset,
    'imagefilelist': pipeline_imagefilelist,
    'imageread': pipeline_imageread,
    'crop': pipeline_crop,
    'resize': pipeline_resize,
    'invert': pipeline_invert,
    'divide_by': pipeline_divide_by,
    'limit': pipeline_limit,
    'order': pipeline_order,
    'normalize_shape': pipeline_normalize_shape,
    'shuffle': pipeline_shuffle
}

def loader(params):
    pipeline = params['pipeline']
    iterator = None
    for op in pipeline:
        name, params = op['name'], op['params']
        iterator = operators[name](iterator, **params)
    return iterator

if __name__ == '__main__':
    from tools.brushstroke.common import disp_grid
    from skimage.io import imsave
    pattern = '{chairs}'
    pipeline = [
        {'name': 'imagefilelist', 'params': {'pattern': pattern}},
        {'name': 'shuffle', 'params': {}},
        {'name': 'limit', "params": {"nb": 10}},
        {'name': 'imageread', 'params': {}},
        {'name': 'normalize_shape', 'params': {}},
        {'name': 'crop', 'params': {'shape': (300, 300), 'pos': 'center', 'mode': 'constant'}},
        {'name': 'resize', "params": {'shape': (64, 64)}},
        {'name': 'divide_by', "params": {"value": 255}},
        #{'name': 'invert', "params": {}},
        {'name': 'order', "params": {'order': 'th'}}
    ]
    iterator = loader({'pipeline': pipeline})
    iterator = minibatch(iterator, batch_size=9)
    iterator = expand_dict(iterator)
    iterator = imap(partial(dict_apply, fn=np.array, cols=['X']), iterator)
    X = (next(iterator)['X'])
    img = disp_grid(X, border=1, bordercolor=(0.3,0,0))
    imsave('out.png', img)
