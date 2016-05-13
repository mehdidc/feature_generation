# -*- coding: utf-8 -*-
"""
This is an implementation of the distance proposed by "An Information Geometry of Statistical Manifold Learning, Ke Sun, Stéphane Marchand-Maillet 
" to measure quality of embedding techniques (e.g PCA, Isomap, LLE, TSNE). It is needs data points matrix of
the original data as well as the same data points with the embedding coordinates, and some additional parameters to fix.

How to use ?
============

Basic swiss roll example:

from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

NN = 100 # nb of  neighbors
k = 10 # logk defines the entropy of tau_Y and tau_Z distributions
ks = 5

Y, labels = make_swiss_roll(1000)

Z = PCA(n_components=2).fit_transform(Y)

dist_to_PCA, _ = compute_dist(Y, Z, NN=NN, k=k, ks=ks, 
                              compute_density=False,
                              nb_integrations=1, # repeat if you want to estimate variance of the integral estimation
                              integration_method='monte_carlo')
T = Isomap(n_components=2).fit_transform(Y)

dist_to_isomap, _ = compute_dist(Y, T, NN=NN, k=k, ks=ks, 
                                compute_density=False,
                                nb_integrations=1, # repeat if you want to estimate variance of the integral estimation
                                integration_method='monte_carlo')
print("dist to PCA : {} +/- {}".format(np.mean(dist_to_PCA), np.std(dist_to_PCA)))
print("dist to Isomap : {} +/- {}".format(np.mean(dist_to_isomap), np.std(dist_to_isomap)))

"""
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import newton
from scipy.integrate import dblquad
from scipy.misc import logsumexp
from itertools import product
from tqdm import tqdm


def softmax(w):
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist
    
def entropy_p_univariate(s):
    r"""
    Compute (carefully to avoid numerical error)
    entropy of p defined by : p_i = exp(-s_i)/(\sum_j exp(-s_j))
    """
    log_norm = logsumexp(-s)
    log_p = - s - log_norm
    ent = (np.exp(log_p) * s).sum() + log_norm
    return ent


def construct_Gcov(S_Y, S_Z, S, NN=5):
    """
    Used to construct 'g_ab' in the paper
    """
    
    #ind = S.argsort(axis=1)
    #ind = ind[:, 1:NN + 1] # start by 1 because we dont want to count a point as neighbor to itself
    ind_Y = S_Y.argsort(axis=1)[:, 1:NN + 1]
    ind_Z = S_Z.argsort(axis=1)[:, 1:NN + 1]
    
    dist_Y = np.zeros((S_Y.shape[0], NN * 2))
    for i in range(S_Y.shape[0]):
        ind = list(set(ind_Y[i]) | set(ind_Z[i]))
        dist_Y[i, 0:len(ind)] = S_Y[i][ind]
        
    dist_Z = np.zeros((S_Y.shape[0], NN * 2))
    for i in range(S_Z.shape[0]):
        ind = list(set(ind_Y[i]) | set(ind_Z[i]))
        dist_Z[i, 0:len(ind)] = S_Z[i][ind]
    
    dist_S = np.empty((S_Y.shape[0], NN * 2))
    dist_S[:, :] = np.inf
    for i in range(S.shape[0]):
        ind = list(set(ind_Y[i]) | set(ind_Z[i]))
        dist_S[i, 0:len(ind)] = S[i][ind]
    
    P_S = softmax(-dist_S)
    Gcov = ((P_S * (dist_Y * dist_Z)).sum(axis=1) - 
            (P_S * dist_Y).sum(axis=1) * (P_S * dist_Z).sum(axis=1))
    return Gcov

def construct_Gvar(S_Y, S, NN=5):
    """
    Used to construct 'g_aa' and 'g_bb' in the paper
    """
    return construct_Gcov(S_Y, S_Y, S, NN=NN)

def construct_tau(S, k, NN=5, tau0=1, verbose=1):
    """
    find tau_i for each i which we scale the distances in row i so that the distribution in row i as entropy logk
    it is done in the paper with 'binary search' or more known as 'bisection method'. the goal of bisection method
    is to find the root of a function. what we want here for a row i is find tau_i so that :
    -\sum_j p_i(j)log p_i(j) = logk
    where p(j) is proba of j given i defined by equation 1 in paper but where the distances
    are scaled by tau_i
    I rather used a newton optimization algo instead of bisection, it works too.
    """
    tau = []
    for i in range(S.shape[0]):
        s = S[i]
        s = s[np.argsort(s)]
        s = s[1:NN + 1]
        def f(tau_sqrt):
            return entropy_p_univariate(s * tau_sqrt**2) - np.log(k)
        tau_sqrt_i = newton(f, tau0, maxiter=100)
        tau_i = tau_sqrt_i ** 2
        if verbose > 0:
            print("row {}, entropy:{}, logk={}, tau:{}".format(i, entropy_p_univariate(s * tau_i), np.log(k), tau_i))
        tau.append(tau_i)
    tau = np.array(tau)
    return tau

def vol(tau_Y, tau_Z, gvar_Y, gvar_Z, gcov_YZ):
    """
    compute the vol term, it is defined in equation 9 of the paper
    """
    cov_term = (tau_Y * tau_Z * gcov_YZ).sum()
    var_Y_term = (tau_Y**2 * gvar_Y).sum()
    var_Z_term = (tau_Z**2 * gvar_Z).sum()
    return np.sqrt(1 - (cov_term**2) / (var_Y_term * var_Z_term))

def sig(tau_Y, tau_Z, gvar_Y, gvar_Z):
    """
    compute the \sigma term, it is defined after proposition 13 in the paper
    """
    var_Y_term = (tau_Y**2 * gvar_Y).sum()
    var_Z_term = (tau_Z**2 * gvar_Z).sum()
    return np.sqrt(var_Y_term) * np.sqrt(var_Z_term)

def compute_density(tau_Y, tau_Z, gvar_Y, gvar_Z, gcov_YZ):
    """
    compute the density term directly (instead of using vol*sig)
    """
    cov_term = (tau_Y * tau_Z * gcov_YZ).sum()
    var_Y_term = (tau_Y**2 * gvar_Y).sum()
    var_Z_term = (tau_Z**2 * gvar_Z).sum()
    return np.sqrt(var_Y_term * var_Z_term - cov_term**2)

def build_density_func(S_Y, S_Z, tau_Y, tau_Z, NN=5):
    def compute(a, b):
        """
        compute density at (a, b). the measure proposed in the paper is a doule integration
        of the density computed in this function for a and b 
        """
        S = a * S_Y + b * S_Z
        #TODO: not correct it should be based on input(Y) OR output(Z) NN neighbours
        gcov_YZ = construct_Gcov(S_Y, S_Z, S, NN=NN) # g_ab in the paper
        gvar_Y = construct_Gvar(S_Y, S, NN=NN) # g_aa in the paper
        gvar_Z = construct_Gvar(S_Z, S, NN=NN) # g_bb in the paper
        return compute_density(tau_Y, tau_Z, gvar_Y, gvar_Z, gcov_YZ)
    return compute

def integrate_2d(func, a_range, b_range, nb_iter=100, verbose=1):
    """
    Generic 2d integration of func(a, b) with monte carlo using uniform sampling for a and b
    
    a_range : tuple defining of the interval to sample a from
    b_range : tuple defining of the interval to sample b from
    """
    val_mean = 0.
    iters = range(nb_iter)
    if verbose > 0:
        iters = tqdm(iters)
    for i in iters:
        a = np.random.uniform(low=a_range[0], high=a_range[1])
        b = np.random.uniform(low=b_range[0], high=b_range[1])
        val = func(a, b)
        val_mean = (val + i * val_mean) / (i + 1)
    return val_mean * (a_range[1] - a_range[0]) * (b_range[1] - b_range[0])

def compute_dist(Y, Z, NN=100, k=10, ks=5, 
                 nb_iter=None, 
                 integration_method='monte_carlo',
                 compute_density=False, 
                 density_grid_size=10, 
                 nb_integrations=1,
                 metric='euclidean',
                 verbose=1):
    S_Y = pairwise_distances(Y, metric=metric)
    S_Z = pairwise_distances(Z, metric=metric)
    tau_Y = construct_tau(S_Y, k=k, tau0=1, NN=NN, verbose=0)
    tau_Z = construct_tau(S_Z, k=k, tau0=1, NN=NN, verbose=0)
    get_density = build_density_func(S_Y, S_Z, tau_Y, tau_Z, NN=NN)
    
    if compute_density:
        a_range = b_range = (1./k, 1./ks)
        a = np.linspace(a_range[0], a_range[1], density_grid_size)
        b = np.linspace(b_range[0], b_range[1], density_grid_size)
        density = np.empty((density_grid_size, density_grid_size))
        for i, j in product(np.arange(len(a)), np.arange(len(b))):
            density[i, j] = get_density(1./a[i], 1./b[j])
    dists = []
    
    a_range = b_range = (ks, k)
    for i in range(nb_integrations):
        if integration_method == 'monte_carlo':
            if nb_iter is None:
                nb_iter = 100
            dist = integrate_2d(get_density, a_range, b_range, nb_iter=nb_iter, verbose=1)
        elif integration_method == 'quad':
            l = 1./k
            r = 1./ks
            dist, err = dblquad(get_density, l, r, lambda a:l, lambda a:r)
            if verbose > 0:
                print('quad integration abs error : {}'.format(err))
        else:
            raise ValueError('Invalid integration method : {}'.format(integration_method))
        dists.append(dist)
    if compute_density:
        return dists, density
    else:
        return dists, None

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def intrinsic_dim_sample_wise(X, k=5):
    neighb = NearestNeighbors(n_neighbors=k).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]
    dist = dist[:, 0:k]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample

def intrinsic_dim_scale_interval(X, k1=10, k2=20):
    X = pd.DataFrame(X).drop_duplicates().values # remove duplicates in case you use bootstrapping
    intdim_k = []
    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(m)
    return intdim_k

def repeated(func, X, nb_iter=100, random_state=None, verbose=0, mode='bootstrap', **func_kw):
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []

    iters = range(nb_iter)
    if verbose > 0:
        iters = tqdm(iters)    
    for i in iters:
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))
    return results