from lightjob.cli import load_db
import joblib
import numpy as np
from joblib import Parallel, delayed
import copy
import pandas as pd

db = load_db()
jobs = db.jobs_with(state='success', type='generation')

def do(j):
    folder =  'jobs/results/{}'.format(j['summary'])
    models = ['tools/models/mnist/m1', 'tools/models/mnist/m2']
    names = ['m1', 'm2']
    stats = j['stats']
    if 'out_of_the_box_classification' not in j['stats']:
        return
    new_stats = copy.deepcopy(stats)
    if 'duration' in new_stats and 'score' in new_stats and 'diversity' in new_stats:
        duration = new_stats['duration']
        score = new_stats['score']
        diversity = new_stats['diversity']
        stats = {'duration': duration, 'score': score, 'diversity': diversity}
        pd.DataFrame(stats).to_csv(folder + '/stats.csv')
        new_stats['duration'] = ''
        new_stats['score'] = ''
        new_stats['diversity'] = ''
    for name, model in zip(names, models):
        s = j['stats']['out_of_the_box_classification'][model]
        if 'predictions' not in s:
            continue
        m = new_stats['out_of_the_box_classification'][model]
        pred = m['predictions']
        new_stats['out_of_the_box_classification'][model] = ''
        new_stats['out_of_the_box_classification'][name] = m
        new_stats['out_of_the_box_classification'][name]['predictions'] = ''
        pred = np.array(pred)
        joblib.dump(pred, "{}/out_of_the_box_classification_{}.npz".format(folder, name), compress=9)
    print('{} completed'.format(j['summary']))
    print(new_stats)
    return new_stats

stats = Parallel(n_jobs=16, verbose=1)(delayed(do)(j) for j in jobs)
for j, s in zip(jobs, stats):
    db.job_update(j['summary'], dict(stats=s))
