import os
from lightjob.cli import load_db
from tqdm import tqdm
from skimage.io import imread, imsave
import json
from collections import OrderedDict
import argparse
import joblib
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__) + '/../..')
from tools.common import disp_grid, to_training
from helpers import mkdir_path
from joblib import Parallel, delayed
import click

def generate_one(j, per_jobset=True):
    id_ = j['summary']
    ref_id_ = j['content']['model_summary']
    img_filename = 'jobs/results/{}/images.npz'.format(id_)
    data = joblib.load(img_filename)
    data = np.array(data)
    if len(data.shape) == 5:
        # (10000, 101, 1, 28, 28)
        data = data[:, -1]
    elif len(data.shape) == 3:
        # (10000, 28, 28)
        data = data[:, None]
    if len(data) == 0:
        return
    print(data.shape)
    data = np.clip(data, 0, 1)
    img = disp_grid(data, border=1, bordercolor=(0.3, 0, .0), normalize=False)
    if per_jobset is False:
        imsave('exported_data/figs/generated/{}.png'.format(id_), img)
    else:
        where_ = db.get_job_by_summary(ref_id_)['where']
        mkdir_path('exported_data/figs/generated/{}'.format(where_))
        imsave('exported_data/figs/generated/{}/{}.png'.format(where_, id_), img)

@click.command()
@click.option('--where', default=None)
@click.option('--per-jobset/--no-per-jobset', default=False)
def generate(where, per_jobset):
    kw = {}
    db = load_db()
    J = db.jobs_with(state='success', type='generation')
    if where:
        ref_jobs = set(map(lambda j:j['summary'], db.jobs_with(where=where)))
        print(ref_jobs)
        J = filter(lambda j:j['content']['model_summary'] in ref_jobs, J)
    print('Nb of jobs : {}'.format(len(J)))
    Parallel(n_jobs=20)(delayed(generate_one)(j, per_jobset=per_jobset) for j in J)

if __name__ == '__main__':
    generate()
