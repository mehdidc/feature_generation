import zlib
from numpy_pickle import read_zfile, ZNDArrayWrapper
import pickle
import numpy as np
import joblib
from tools.common import disp_grid
from skimage.io import imsave
import shutil
data = joblib.load(('/home/mcherti/work/code/feature_generation/jobs/results/f237f4fdfc2f6415f9e1ae0681162bef/images.npz'))
_, init_args , state = data.__reduce__()
summaries = [
#    'd92bd8ae6b052fddee51c323fce60b61',
#    '6ad5769819aa6f86c846b50990072e74',
#    '7de0531049c360bbe3917e081fceaa61',
#    '0c35d5aad2c9d1abb8a589c821636f57',
#    '31d3fb4c4559e5a73b4b1370b981e50a',
#    'c5b1e8f5bba7c84a5dccf8c0150db50a',
#    'a38a2774208ea4450a784d598f7a3bbc',
#    '0f8a5d9ab3df23f7d7faf1835cab9a68',
#    'bfa03e7f7701d21e3079790a55ebe754',
#    '417155804712b1fe17c8eb80094862a9',
#    '48fd385467bf47bc5eefa485581331fe',
]
for s in summaries:
    zfile = '/home/mcherti/work/code/feature_generation/jobs/results/{}/images.npz_01.npy.z'.format(s)
    st = read_zfile(open(zfile))
    shutil.copy(zfile, zfile + '.bak')
    data.__setstate__(state[0:-1] + (st,))
    joblib.dump(data, '/home/mcherti/work/code/feature_generation/jobs/results/{}/images.npz'.format(s), compress=9)
    d = data[:, -1]
    print(d.shape)
    img = disp_grid(d)
    imsave('out.png', img)
