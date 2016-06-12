import sys
import os
sys.path.append(os.path.dirname(__file__)+"/..")
from sklearn.cluster import MeanShift
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
import intdim_mle
import manifold

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def genstats(jobs, db, force=False, n_jobs=-1, filter_stats=None):
    stats = Parallel(n_jobs=n_jobs)(delayed(compute_stats)(dict(j), force=force, filter_stats=filter_stats)
                                    for j in jobs)
    for j, s in zip(jobs, stats):
        update_stats(j, s, db)


def update_stats(job, stats, db):
    db.job_update(job["summary"], dict(stats=stats))


def compute_stats(job, force=False, filter_stats=None):
    j = job
    folder = "jobs/results/{}".format(j['summary'])
    hash_matrix_filename = os.path.join(folder, "csv", "hashmatrix.npy")
    hash_matrix = np.load(hash_matrix_filename)
    x = hash_matrix_to_int(hash_matrix)
    stats = j.get("stats", {})
    if stats is None:
        stats = {}
    s = job['summary']
    ref_job = j['ref_job']['summary']

    if filter_stats is not None:
        filter_stats = set(filter_stats.split(','))
    def should_compute(s, stats):
        if filter_stats is None:
            if force:
                return True
            if s in stats:
                return False
            return True
        else:
            if s not in filter_stats:
                return False
            if force:
                return True
            if s in stats:
                return False
            return True
    if should_compute('mean', stats):
        logger.info('compute mean of {}'.format(s))
        stats["mean"] = x.mean()
        stats['mean'] = stats['mean'] / len(hash_matrix)
    if should_compute('var', stats):
        logger.info('compute var of {}'.format(s))
        stats["var"] = x.var(ddof=1)
        maxvar = ((10000 - 1 + 1)**2 - 1) / 12# variance of max entropy distribution (discrete uniform between 1 and 10000)
        stats['var'] = stats['var'] / maxvar

    if should_compute('skew', stats):
        logger.info('compute skew of {}'.format(s))
        stats["skew"] = skew(x)

    if should_compute('multiplecorrelation', stats):
        logger.info('compute multiplecorrelation of {}'.format(s))
        stats["multiplecorrelation"] = compute_neighcorr(folder, hash_matrix, formula='multiplecorrelation')

    if should_compute('clusdiversity', stats):
        logger.info('computing diversity score using clustering of {}'.format(s))
        stats["clusdiversity"] = compute_clusdiversity(folder, hash_matrix)
        maxdist = np.sqrt(784)# we have binary images, so euclidean dist between full zero vector minus full one vector
        stats["clusdiversity"] = stats["clusdiversity"] / maxdist

    if should_compute('intdim_mle', stats):
        logger.info('computing intdim_le of {}'.format(s))
        stats["intdim_mle"] = compute_intdim(folder, hash_matrix, method='mle')

    if should_compute('convergence_speed', stats):
        logger.info('computing convergence speed of {}'.format(s))
        stats['convergence_speed'] = compute_convergence_speed(folder, j)

    if should_compute('fonts_rec_error', stats):
        logger.info('computing reconstruction error on fonts'.format(s))
        stats['fonts_rec_error'] = compute_rec_error(j, 'fonts', ref_job)

    if should_compute('fontness', stats):
        logger.info('compute fontness of generated data')
        scores = compute_modelness(folder, j, 'discriminators/fonts_32x32.pkl')
        stats['fontness'] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            '10per': float(np.percentile(scores, 10)),
            '90per': float(np.percentile(scores, 90))
        }

    #if "nb_almost_uniques" not in stats or force:
    #    logger.info('computing nb  of almost uniques of {} (with meanshift)'.format(s))
    #    stats['nb_almost_uniques'] = compute_almost_uniq(folder, hash_matrix)

    #if "manifold_dist" not in stats or force:
    #    logger.info('computing manifold distance of {}'.format(s))
    #    stats['manifold_dist'] = compute_manifold_dist(folder, hash_matrix, ref_job)
    #if "nearestneighborsdiversity" not in stats or force:
    #    logger.info('computing nearest neighbors diversity score using clustering of {}'.format(s))
    #    stats["nearestneighborsdiversity"] = compute_nearestneighbours_diversity(folder, hash_matrix)
    #if "dpgmmnbclus" not in stats or force:
    #    logger.info('computing nb of clusters with DPGMM of {}'.format(s))
    #    stats["dpgmmnbclus"] = compute_dpgmmclusdiversity(folder, hash_matrix)
    logger.info('Finished on {}, stats : {}'.format(s, stats))
    return stats


def construct_data(job_folder, hash_matrix, transform=lambda x:x):
    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    filenames = sorted(filenames)
    indices = unique_indices(hash_matrix)
    filenames = [filenames[ind] for ind in indices]
    if len(filenames) == 0:
        return None

    X = []
    for im in get_images(filenames):
        X.append([transform(im)])
    X = np.concatenate(X, axis=0)
    X = X.reshape((X.shape[0], -1))
    return X

def compute_modelness(job_folder, hash_matrix, model_path):
    import pickle
    from lasagne import layers
    import theano.tensor as T
    from skimage.transform import resize
    import theano
    print('load data...')
    data = pickle.load(open(model_path))
    discr = data['discriminator']
    discr_weights = data['discriminator_weights']
    layers.set_all_param_values(discr, discr_weights)

    print('compile function...')
    X = T.tensor4()

    get_score = theano.function([X], layers.get_output(discr,X))

    input_layer = layers.get_all_layers(discr)[0]
    def transform(x):
        c, w, h = input_layer.output_shape[1:]
        if x.shape[1] != w or x.shape[2] != h:
            return resize(x, (w, h), preserve_range=True)
        else:
            return x
    print('create data...')
    imgs = construct_data(job_folder, hash_matrix, transform=transform)
    imgs -= imgs.min()
    imgs /= imgs.max()
    print(imgs.max(), imgs.min())
    shape = (imgs.shape[0],) + input_layer.output_shape[1:]
    imgs = imgs.reshape(shape)
    imgs = imgs.astype(np.float32)
    print('compute scores...')
    scores = minibatcher(imgs, get_score, size=1000)[:, 0]
    return scores


def minibatcher(X, f, size=128):
    from lasagnekit.easy import iterate_minibatches
    res = []
    for sl in iterate_minibatches(X.shape[0], size):
        r = f(X[sl])
        res.append(r)
    return np.concatenate(res, axis=0)

def compute_rec_error(job, dataset, ref_job):
    from tasks import check
    from data import load_data
    from lasagnekit.easy import iterate_minibatches

    v = check(what="notebook",
              filename="jobs/results/{}/model.pkl".format(ref_job),
              dataset='digits') # any would work, we dont care
    capsule, data, layers, w, h, c = v
    data = load_data(dataset, w=w, h=h)
    assert hasattr(data, 'train')
    X = data.train.X
    rec_errors = []
    for batch in iterate_minibatches(X.shape[0], batchsize=1000):
        rec_error = ((capsule.preprocess(X[batch]) - capsule.reconstruct(capsule.preprocess(X[batch])))**2).mean(axis=(1, 2, 3))
        rec_errors.append(rec_error)
    return float(np.concatenate(rec_errors, axis=0).mean())

def compute_convergence_speed(job_folder, job):
    max_nb_iterations = 100.
    speed = 1. - (len(open(os.path.join(job_folder, "csv", "iterations.csv")).readlines()) - 1) / max_nb_iterations
    # speed = 0 (worst) 1(best)
    return speed

def compute_almost_uniq(folder, hash_matrix):
    bandwidth = None # meaning it is estimated from data
    X = construct_data(folder, hash_matrix)
    try:
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(X)
        return len(ms.cluster_centers_) / float(len(X))
    except Exception:
        return np.nan


def compute_manifold_dist(job_folder, hash_matrix, ref_job):

    import numpy as np
    from tasks import check
    import theano
    import theano.tensor as T
    import lasagne

    v = check(what="notebook",
              filename="jobs/results/{}/model.pkl".format(ref_job),
              dataset="digits")
    capsule, data, layers, w, h, c = v

    np.random.seed(1234)
    nb = 10000
    print(data.train.X.shape)
    Y = data.train.X[np.random.choice(np.arange(len(data.train.X)), size=nb, replace=False)]
    x = T.tensor4()
    g = theano.function(
        [x],
        lasagne.layers.get_output(layers['hid'], x)
    )
    H = g(Y.reshape((Y.shape[0], c, w,h)))

    NN = 100 # nb of  neighbors
    k = 10 # logk defines the entropy of tau_Y and tau_Z distributions
    ks = 5
    dist, _ = manifold.compute_dist(Y, H, NN=NN, k=k, ks=ks,
                                    compute_density=False,
                                    nb_integrations=1, # repeat if you want to estimate variance of the integral estimation
                                    nb_iter=100,
                                    integration_method='monte_carlo')
    return np.mean(dist)

def compute_dpgmmclusdiversity(job_folder, hash_matrix):
    from sklearn.mixture import DPGMM
    X = construct_data(job_folder, hash_matrix)
    clus = DPGMM(n_components=1000, verbose=1)
    clus.fit(X)
    n_components = clus.n_components
    print('Nb of components : ', n_components)
    return n_components


def compute_clusdiversity(job_folder, hash_matrix, nb_clusters=1000):
    from sklearn.cluster import KMeans
    X = construct_data(job_folder, hash_matrix)
    if X.shape[0] < nb_clusters:
        nb_clusters = max(X.shape[0] / 10, 1)
    logger.info(str(nb_clusters))
    clus = KMeans(n_clusters=nb_clusters, verbose=1, n_jobs=1, n_init=2)
    clus.fit(X)
    # dists from closest cluster center
    dists = clus.transform(X).min(axis=1)
    # avg distance of points to their closest cluster center
    return dists.mean()

def compute_intdim(job_folder, hash_matrix, method='mle'):
    import numpy as np
    np.random.seed(42)
    X = construct_data(job_folder, hash_matrix)
    nb = min(100, X.shape[0])
    X = X[np.random.choice(np.arange(len(X)), size=nb, replace=False)]
    k1 = 10 # start of interval(included)
    k2 = 20 # end of interval(included)
    intdim_k_repeated = intdim_mle.repeated(
        intdim_mle.intrinsic_dim_scale_interval,
        X,
        mode='bootstrap',
        nb_iter=100, # nb_iter for bootstrapping
        verbose=1,
        k1=k1, k2=k2)
    print('Estimate : {} +/- {}'.format(np.mean(np.mean(intdim_k_repeated, axis=1)), np.var(np.mean(intdim_k_repeated, axis=1))))
    return np.mean(intdim_k_repeated)

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
