import os
import sys
sys.path.append(os.path.dirname(__file__) + '/../..')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.io import imread

from lightjob.cli import load_db

from tqdm import tqdm

from data import load_data

import cPickle as pickle


def get_data(filename):
    fd = open(filename, "r")
    data = pickle.load(fd)
    fd.close()
    return data


def plot_dataset(code_2d, categories):
    colors = [
        'r',
        'b',
        'g',
        'crimson',
        'gold',
        'yellow',
        'maroon',
        'm',
        'c',
        'orange'
    ]
    for cat in range(0, 10):
        g = categories == cat
        plt.scatter(code_2d[g, 0], code_2d[g, 1],
                    marker='+', c=colors[cat], s=40, alpha=0.7,
                    label="digit {}".format(cat))


def plot_generated(code_2d, categories):
    g = categories < 0
    plt.scatter(code_2d[g, 0], code_2d[g, 1], marker='+',
                c='gray', s=30)

if __name__ == '__main__':
    db = load_db()

    J = db.jobs_with(state='success', type='generation')

    dataset = load_data(dataset='digits')
    dataset = dataset.train

    for j in tqdm(J):
        id_ = j['summary']
        jref_s = j['content']['model_summary']
        jref = db.get_job_by_summary(jref_s)
        filename = 'jobs/results/{}/tsne_input.pkl'.format(id_)
        if not os.path.exists(filename):
            continue
        img_filename = 'jobs/results/{}/final1000.png'.format(id_)
        if not os.path.exists(img_filename):
            continue
        img_content = imread(img_filename)
        data = get_data(filename)
        for c in ('x', 'y'):
            data[c] = (data[c] - data[c].mean()) / data[c].std()

        fig = plt.figure(figsize=(15, 22))

        plt.subplot(2, 1, 1)
        #print(data['x'].shape, data['y'].shape, data['is_generated'].shape, data['dataset_ind'].shape, data['gen_ind'].shape)
        code_2d = pd.DataFrame({'x': data['x'], 'y': data['y']}).values
        cats = data['is_generated'].astype(int)
        cats[cats == 1] = -1
        cats[cats == 0] = dataset.y[data['dataset_ind']]
        try:
            plot_dataset(code_2d, cats)
            plot_generated(code_2d, cats)
        except Exception:
            continue

        plt.title(id_+'/'+jref_s)
        plt.subplot(2, 1, 2)
        plt.imshow(img_content, cmap='gray', interpolation='none')
        plt.title(id_+'/'+jref_s)
        plt.savefig('figs/tsne/{}.png'.format(id_))
        plt.show()
        plt.close(fig)
