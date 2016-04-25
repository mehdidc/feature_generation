import os
import json
from collections import defaultdict, OrderedDict
import os
import pandas as pd
import matplotlib as mpl
import glob
from collections import Counter

def update_stats(job, db):
	stats = compute_stats(jobs)
	db.job_update(job["summary"], dict(stats=stats))

def compute_stats(job):
	j = job
	folder = "jobs/results/{}".format(j['summary'])
    hash_matrix_filename = os.path.join(folder, "csv", "hashmatrix.npy")
	hash_matrix = np.load(hash_matrix_filename)
	x = hash_matrix_to_int(hash_matrix)
	stats = j.get("stats", {})
	if "mean" not in stats:
		stats["mean"] = x.mean()
	if "var" not in stats:
		stats["var"] = x.var(ddof=1)
	if "skew" not in stats:
		stats["skew"] = skew(x)
	if "nc" not in stats:
		stats["neighcorr"] = compute_neighcorr(folder, hash_matrix)
	return stats

def compute_neighcorr(job_folder, hash_matrix):
    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    filenames = sorted(filenames)
   	K = hash_matrix_to_indices(hash_matrix)
    indices = K.values()
    filenames = [filenames[ind] for ind in indices]
    print('comute neigh in {}'.format(len(filenames)))
    nc = neighbcorr_filenames(filenames)
    return nc

def neighbcorr_filenames(filenames):
    from skimage.io import imread
    corr = []
    corrdata = defaultdict(int)
    for i, f in enumerate(filenames):
        im = imread(f)
        im = 2 * (im / im.max()) - 1
        assert set(im.flatten().tolist()) <= set([1, -1]), set(im.flatten().tolist())
        neighcorr(im, corrdata=corrdata, pad=3)
    nc = np.abs(np.array(corrdata.values())).mean() / len(filenames)
    return nc

def neighcorr(im, corrdata=None, pad=3):
    assert corrdata is not None
    i = 0
    for x in range(pad, im.shape[0] - pad):
        for y in range(pad, im.shape[1] - pad):
            pxc = im[x, y]
            ctot = 0
            for dx, dy in product((0, 1, -1), (0, 1, -1)):
                px = im[x+dx, y+dy]
                c = px * pxc
                corrdata[i] += c
            i += 1

def hash_matrix_to_int(hm):
    cnt = Counter(hm)
    s = np.argsort(cnt.values())[::-1]
    K = cnt.keys()
    K = [K[s[i]] for i in range(len(K))]
    K_to_int = {k: i + 1 for i, k in enumerate(K)}
    x = [K_to_int[v] for v in hm]
    x = np.array(x)
    return x

def hash_matrix_to_indices(hm):
	pass


