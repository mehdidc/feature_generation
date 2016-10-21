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
from tools.common import disp_grid
from helpers import mkdir_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--per_jobset', default=1, type=int, required=False)
    parser.add_argument('--where', default=None, type=str, required=False)

    args = parser.parse_args()
    per_jobset = args.per_jobset == 1
    where = args.where
    kw = {}
    db = load_db()
    J = db.jobs_with(state='success', type='generation')
    if where:
        ref_jobs = set(map(lambda j:j['summary'], db.jobs_with(where=where)))
        print(ref_jobs)
        J = filter(lambda j:j['content']['model_summary'] in ref_jobs, J)
    print('Nb of jobs : {}'.format(len(J)))
    for j in tqdm(J):
        id_ = j['summary']
        ref_id_ = j['content']['model_summary']
        img_filename = 'jobs/results/{}/images.npz'.format(id_)
        data = joblib.load(img_filename)
        data = np.array(data)
        data = data[:, -1]
        data = np.clip(data, 0, 1)
        img = disp_grid(data, border=1, bordercolor=(0.3, 0, .0), normalize=False)
        if per_jobset is False:
            imsave('exported_data/figs/generated/{}.png'.format(id_), img)
        else:
            mkdir_path('exported_data/figs/generated/{}'.format(where))
            imsave('exported_data/figs/generated/{}/{}.png'.format(where, id_), img)
