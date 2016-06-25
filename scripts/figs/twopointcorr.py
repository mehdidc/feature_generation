import matplotlib as mpl
mpl.use('Agg')
import sys
import numpy as  np
import os
import matplotlib.pyplot as plt
import pandas as pd
from lightjob.cli import load_db
from astroML.correlation import two_point
from sklearn.metrics import euclidean_distances
from sklearn.manifold import TSNE
from tqdm import tqdm
from joblib import Memory

from tempfile import mkdtemp
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)

np.random.seed(2)

db = load_db()

J = db.jobs_with(state='success', type='generation')

@memory.cache
def build_baseline(nb=1000):
    R = np.random.uniform(size=(X.shape[0], 784))#TODO change, only tied to mnist
    tsne = TSNE(perplexity=30, early_exaggeration=4., verbose=0, n_components=2)
    R = tsne.fit_transform(R)
    return R


for j in tqdm(J):
    id_ = j['summary']
    jref_s = j['content']['model_summary']
    jref = db.get_job_by_summary(jref_s)
    filename = 'jobs/results/{}/tsne_input.csv'.format(id_)
    if not os.path.exists(filename):
        continue
    df = pd.read_csv(filename)
    X = df[['x', 'y']].values
    dist = euclidean_distances(X)
    dist_min, dist_max = dist.min(), dist.max()
    nb = 30
    width = (dist_max - dist_min) / nb
    c = 0
    bins = np.linspace(dist_min - c, dist_max + c, nb)
    R = build_baseline(nb=X.shape[0])
    t = two_point(X, bins, method='landy-szalay', data_R=R)

    bins = (bins - bins.min()) / (bins.max() - bins.min())
    pt = (bins[0:-1] + bins[1:])/2.
    fig = plt.figure(figsize=(17, 5))
    plt.xticks(pt, fontsize=8)
    plt.plot(pt, t)
    plt.scatter(pt, t)
    plt.axhline(0, linestyle='dashed', c='gray')
    for b in bins:
        plt.axvline(x=b, color='gray', linestyle='dashed')
    plt.xlabel('normalized distance')
    plt.ylabel('two point correlation')
    plt.savefig('figs/twopointcorr/{}'.format(id_))
    plt.close(fig)
