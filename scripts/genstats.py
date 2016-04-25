\

import os
import json
from collections import defaultdict, OrderedDict
from itertools import product
import os
import pandas as pd
import matplotlib as mpl
import glob
import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from scipy.stats import skew
import pickle
import sys
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def genstats(jobs, db, force=False, n_jobs=-1):
    stats = Parallel(n_jobs=n_jobs)(delayed(compute_stats)(dict(j), force=force)
                                    for j in jobs)
    for j, s in zip(jobs, stats):
        update_stats(j, s, db)


def update_stats(job, stats, db):
    db.job_update(job["summary"], dict(stats=stats))


def compute_stats(job, force=False):
    j = job
    folder = "jobs/results/{}".format(j['summary'])
    hash_matrix_filename = os.path.join(folder, "csv", "hashmatrix.npy")
    hash_matrix = np.load(hash_matrix_filename)
    x = hash_matrix_to_int(hash_matrix)
    stats = j.get("stats", {})
    s = job['summary']
    if "mean" not in stats or force:
        logger.info('compute mean of {}'.format(s))
        stats["mean"] = x.mean()
    if "var" not in stats or force:
        logger.info('compute var of {}'.format(s))
        stats["var"] = x.var(ddof=1)
    if "skew" not in stats or force:
        logger.info('compute skew of {}'.format(s))
        stats["skew"] = skew(x)
    #if "neighcorr" not in stats or force:
    #    logger.info('compute neighcorr of {}'.format(s))
    #    stats["neighcorr"] = compute_neighcorr(folder, hash_matrix, formula='mine')
    if "multiplecorrelation" not in stats or force:
        logger.info('compute multiplecorrelation of {}'.format(s))
        stats["multiplecorrelation"] = compute_neighcorr(folder, hash_matrix, formula='multiplecorrelation')
    if "clusdiversity" not in stats or force:
        logger.info('computing diversity score using clustering of {}'.format(s))
        stats["clusdiversity"] = compute_clusdiversity(folder, hash_matrix)
    #if "nearestneighborsdiversity" not in stats or force:
    #    logger.info('computing nearest neighbors diversity score using clustering of {}'.format(s))
    #    stats["nearestneighborsdiversity"] = compute_nearestneighbours_diversity(folder, hash_matrix)
    #if "dpgmmnbclus" not in stats or force:
    #    logger.info('computing nb of clusters with DPGMM of {}'.format(s))
    #    stats["dpgmmnbclus"] = compute_dpgmmclusdiversity(folder, hash_matrix)
    logger.info('Finished on {}, stats : {}'.format(s, stats))
    return stats


def compute_dpgmmclusdiversity(job_folder, hash_matrix):
    from sklearn.mixture import DPGMM

    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    filenames = sorted(filenames)
    indices = unique_indices(hash_matrix)
    filenames = [filenames[ind] for ind in indices]
    if len(filenames) == 0:
        return None
    
    X = []
    for im in get_images(filenames):
        X.append([im])
    X = np.concatenate(X, axis=0)
    X = X.reshape((X.shape[0], -1))


    clus = DPGMM(n_components=1000, verbose=1)
    clus.fit(X)
    n_components = clus.n_components
    print('Nb of components : ', n_components)
    return n_components


def compute_clusdiversity(job_folder, hash_matrix, nb_clusters=1000):
    from sklearn.cluster import KMeans

    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    filenames = sorted(filenames)
    indices = unique_indices(hash_matrix)
    filenames = [filenames[ind] for ind in indices]
    if len(filenames) == 0:
        return None
    X = []
    for im in get_images(filenames):
        X.append([im])
    X = np.concatenate(X, axis=0)
    X = X.reshape((X.shape[0], -1))

    if X.shape[0] < nb_clusters:
        nb_clusters = max(X.shape[0] / 10, 1)
    logger.info(str(nb_clusters))
    clus = KMeans(n_clusters=nb_clusters, verbose=1, n_jobs=1, n_init=2)
    clus.fit(X)
    # dists from closest cluster center
    dists = clus.transform(X).min(axis=1)
    # avg distance of points to their closest cluster center
    return dists.mean()


def compute_nearestneighbours_diversity(job_folder, hash_matrix, nearest_neighbors=100):
    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    X = []
    for im in get_images(filenames):
        X.append([im])
    X = np.concatenate(X, axis=0)
    X = X.reshape((X.shape[0], -1))
    dist = nearestneighbours_distance(X, nearest_neighbors=nearest_neighbors)
    return dist.mean()



def nearestneighbours_distance(X, nearest_neighbors=8):
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=nearest_neighbors, algorithm='ball_tree').fit(X)
        distances, _ = nbrs.kneighbors(X)
        return distances.mean(axis=1)


def compute_neighcorr(job_folder, hash_matrix, formula='mine'):
    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    filenames = sorted(filenames)
    indices = unique_indices(hash_matrix)
    filenames = [filenames[ind] for ind in indices]
    if formula == 'mine':
        nc = neighbcorr_filenames_mine(filenames)
    elif formula == 'multiplecorrelation':
        nc = neighbcorr_filenames_multiplecorrelation(filenames)
    return nc


def unique_indices(hm):
    K = {}
    for i, h in enumerate(hm):
        if h not in K:
            K[h] = i
    return K.values()


def neighbcorr_filenames_multiplecorrelation(filenames, pad=3):
    # https://en.wikipedia.org/wiki/Multiple_correlation
    from sklearn.linear_model import LinearRegression
    xdata = defaultdict(list)
    ydata = defaultdict(list)
    for im in get_images(filenames):
        for x in range(pad, im.shape[0] - pad):
            for y in range(pad, im.shape[1] - pad):
                pxc = im[x, y]
                dt = []
                for dx, dy in product((0, 1, -1), (0, 1, -1)):
                    if dx == 0 and dy == 0:
                        continue
                    px = im[x + dx, y + dy]
                    dt.append(px)
                xdata[(x, y)].append(dt)
                ydata[(x, y)].append(pxc)
    rsqr_l = []
    for pos in xdata.keys():
        x = xdata[pos]
        y = ydata[pos]
        rsqr = np.sqrt(LinearRegression().fit(x, y).score(x, y))
        rsqr_l.append(rsqr)
    return np.mean(rsqr_l)


def neighbcorr_filenames_mine(filenames):
    corrdata = defaultdict(int)
    for im in get_images(filenames):
        neighcorr(im, corrdata=corrdata, pad=3)
    nc = np.abs(np.array(corrdata.values())).mean() / len(filenames)
    return nc


def get_images(filenames):
    from skimage.io import imread
    for f in filenames:
        im = imread(f)
        im = 2 * (im / im.max()) - 1
        assert set(im.flatten().tolist()) <= set([1, -1]), set(im.flatten().tolist())
        yield im


def neighcorr(im, corrdata=None, pad=3):
    assert corrdata is not None
    i = 0
    for x in range(pad, im.shape[0] - pad):
        for y in range(pad, im.shape[1] - pad):
            pxc = im[x, y]
            for dx, dy in product((0, 1, -1), (0, 1, -1)):
                px = im[x + dx, y + dy]
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
