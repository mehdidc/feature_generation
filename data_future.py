from itertools import imap
from functools import partial
import random
import glob

from skimage.transform import resize
from skimage.util import pad
from skimage.io import imread

import numpy as np

import datakit
from datakit.helpers import dict_apply, minibatch, expand_dict, data_path, ncycles

dataset_patterns = {
    'sketchy': 'sketchy/256x256/sketch/tx_000000000000/**/*.png',
    'omniglot': 'omniglot/**/**/**/*.png',
    'flaticon': 'flaticon/**/png/*.png',
    'lfw': 'lfw/imgaligned/**/*.jpg',
    'shoes': 'shoes/ut-zap50k-images/Shoes/**/**/*.jpg',
    'svhn': 'svhn/**/*.png',
    'chairs': 'chairs/rendered_chairs/**/renders/*.png',
    'icons': 'icons/img/*.png',
    'aloi': 'aloi/png4/**/*.png',
    'kanji': 'kanji/cleanpngsmall/*.png',
    'iam': 'iam/**/**/*.png',
    'yale': 'yale/YALE/**/*.pgm',
    'yale_b': 'yale_b/**/**/*.pgm',
    'eyes': 'eyes/**/**/*.png',
    'gametiles': 'gametiles/zw-tilesets/img/*.png',
    'faces94': 'faces94/**/**/*.jpg',
    'dlibfaces': 'dlibfaces/dlib_face_detection_dataset/**/**/*.png'
}

def apply_to(fn, cols=None):
    def fn_(iterator, *args, **kwargs):
        iterator = imap(partial(dict_apply, fn=partial(fn, *args, **kwargs), cols=cols), iterator)
        return iterator
    return fn_

def crop(img, shape=(1, 1), pos='random', mode='constant', rng=np.random):
    # assumes img is shape (h, w, color)
    assert len(img.shape) == 3 and img.shape[2] in (1, 3)
    img_h, img_w, img_c = img.shape
    h, w = shape
    if pos == 'random':
        y = rng.randint(0, img_h - 1)
        x = rng.randint(0, img_w - 1)
    elif pos == 'random_inside':
        y = rng.randint(h, img_h - h//2 - 1)
        x = rng.randint(w, img_w - w//2 - 1)
    elif pos == 'center':
        y = img_h // 2
        x = img_w // 2
    else:
        raise Exception('Unkown mode')
    out_img = np.empty((h, w, img_c))
    img_ = np.zeros((img_h + h, img_w + w, img_c))
    for c in range(img_c):
        img_[:, :, c] = pad(img[:, :, c], (h//2, w//2), str(mode))
    img = img_[y:y+h, x:x+w, :]
    return img

def order(X, order='th'):
    if order == 'th':
        X = X.transpose((2, 0, 1))
    elif order == 'tf':
        X = X.transpose((1, 2, 0))
    return X

def resize_(X, shape=(1,1)):
    X = resize(X, output_shape=shape, preserve_range=True)
    return X

def invert(X):
    return 1 - X

def divide_by(X, value=255.):
    return X / value

def normalize_shape(X):
    # if shape = 2, add a new axis at the right
    # if shape = 3, leave it as it is
    if len(X.shape) == 2:
        X = X[:, :, np.newaxis]
    if X.shape[2] > 3:
        # if alpha channel, remove it
        X = X[:, :, 0:-1]
    return X

def force_rgb(X):
    # if 1 channel, force to have 3 channels
    if X.shape[2] == 1:
        X = X * np.ones((1, 1, 3))
    return X

colors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1 ,0],
    [0, 1, 1],
    [1, 0, 1]
]
colors = np.array(colors, dtype='float32') * 255
def random_colorize(X, foreground=128, op='threshold', rng=np.random):
    if X.shape[2] == 1:
        if op == 'threshold':
            foreground_mask = (X > foreground)
        elif op == 'threshold_inv':
            foreground_mask = (X <= foreground)
        else:
            raise Exception('Unknown op : {}'.format(op))
        #col = np.random.uniform(size=3)
        col = colors[rng.randint(0, len(colors) - 1)]
        foreground_color = 255 - col
        #foreground_color = np.random.uniform(size=3)
        X_new = np.ones((X.shape[0], X.shape[1], 3)) * col[np.newaxis, np.newaxis, :]
        X_new = X_new * (1 - foreground_mask) + foreground_mask * foreground_color
        X = X_new
    return X

pipeline_crop = apply_to(crop, cols=['X'])
pipeline_order = apply_to(order, cols=['X'])
pipeline_resize = apply_to(resize_, cols=['X'])
pipeline_invert = apply_to(invert, cols=['X'])
pipeline_divide_by = apply_to(divide_by, cols=['X'])
pipeline_normalize_shape = apply_to(normalize_shape, cols=['X'])
pipeline_force_rgb = apply_to(force_rgb, cols=['X'])
pipeline_random_colorize = apply_to(random_colorize, cols=['X'])
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

def pipeline_repeat(iterator, nb=1):
    return ncycles(iterator, n=nb)

def pipeline_load_dataset(iterator, name, *args, **kwargs):
    assert hasattr(datakit, name)
    module = getattr(datakit, name)
    return module.load_as_iterator(*args, **kwargs)

operators = {
    'dataset': pipeline_load_dataset,
    'random_colorize': pipeline_random_colorize,
    'imagefilelist': pipeline_imagefilelist,
    'imageread': pipeline_imageread,
    'crop': pipeline_crop,
    'resize': pipeline_resize,
    'invert': pipeline_invert,
    'divide_by': pipeline_divide_by,
    'limit': pipeline_limit,
    'order': pipeline_order,
    'normalize_shape': pipeline_normalize_shape,
    'shuffle': pipeline_shuffle,
    'repeat': pipeline_repeat,
    'force_rgb': pipeline_force_rgb
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
    import numpy as np
    import random

    params = {
        "pipeline": [
            {"name": "dataset", "params": {"name": "mnist", "which":"train"}},
            {"name": "order", "params": {"order": "tf"}},
            {"name": "shuffle", "params": {}},
            {"name": "normalize_shape", "params": {}},
            {"name": "random_colorize", "params":{"op": "threshold_inv"}},
            {"name": "resize", "params": {"shape": [16, 16]}},
            {"name": "force_rgb", "params": {}},
            {"name": "divide_by", "params": {"value": 255}},
            {"name": "order", "params": {"order": "th"}}
        ]
    }
    random.seed(10)
    np.random.seed(10)
    iterator = loader(params)
    iterator = minibatch(iterator, batch_size=1000)
    iterator = expand_dict(iterator)
    iterator = imap(partial(dict_apply, fn=np.array, cols=['X']), iterator)
    i = 0
    for data in iterator:
        X = data['X']
        i += len(X)
        print(i)
        break
    img = disp_grid(X, border=1, bordercolor=(0.3,0,0))
    imsave('out.png', img)
