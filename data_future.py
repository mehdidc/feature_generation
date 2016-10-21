from itertools import imap
from functools import partial
import random
import glob

from skimage.transform import resize
from skimage.util import pad
from skimage.io import imread

import numpy as np

from datakit.image import pipeline_load as loader
from datakit.image import operators

import h5py

def pipeline_load_hdf5(iterator, filename, cols=['X']):
    hf = h5py.File(filename)
    X = hf['X']
    return X

operators['load_hdf5'] = pipeline_load_hdf5
loader = partial(loader, operators=operators)

if __name__ == '__main__':
    from datakit.helpers import minibatch, expand_dict, dict_apply
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
    iterator = loader(params['pipeline'])
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
