import numpy as np
from lightjob.cli import load_db
from lightjob.db import SUCCESS
from lightjob.utils import dict_format
import pandas as pd
import sys
from hp_toolkit.helpers import flatten_dict
sys.path.append('.')
from tools.common import to_generation

def dict_concat(d1, d2):
    d = dict()
    d.update(d1)
    d.update(d2)
    return d

def get_hypers(jobs):
    J = jobs
    jobs = [dict_concat(j['content'], {'id': j['summary'], 'jobset': j['where']}) for j in jobs]
    jobs = map(flatten_dict, jobs)
    cols = set([c for j in jobs for c in j.keys()])
    scores = {

       'objectness': 'stats.out_of_the_box_classification.letterness.objectness',

       #'in_objectness': 'stats.out_of_the_box_classification.m2.objectness',
       #'in_count': 'stats.out_of_the_box_classification.letterness.diversity_count_digits_95',
       #'in_max':'stats.out_of_the_box_classification.letterness.diversity_max_digits',
       #'out_objectness': 'stats.out_of_the_box_classification.fonts.objectness',
       #'out_count': 'stats.out_of_the_box_classification.letterness.diversity_count_letters_95',
       #'out_max':'stats.out_of_the_box_classification.letterness.diversity_max_letters',

        'in_objectness': 'stats.out_of_the_box_classification.m2.sample_objectness',
        'in_count': 'stats.out_of_the_box_classification.letterness.count_digits_95',
        'in_max': 'stats.out_of_the_box_classification.letterness.max_digits',
        'in_diversity': 'stats.out_of_the_box_classification.letterness.diversity_digits',
        'diversity': 'stats.out_of_the_box_classification.letterness.sample_objectness',
        'out_objectness': 'stats.out_of_the_box_classification.fonts.sample_objectness',
        'out_count': 'stats.out_of_the_box_classification.letterness.count_letters_95',
        'out_max':'stats.out_of_the_box_classification.letterness.max_letters',
        'out_diversity': 'stats.out_of_the_box_classification.letterness.diversity_letters',
    }
    inputs = pd.DataFrame(jobs)
    for name, field in scores.items():
        inputs[name] = [dict_format(j, field, if_not_found=np.nan) for j in J]
    inputs['inout_noise'] = 1 -inputs['in_count'] - inputs['out_count']
    return inputs

if __name__ == '__main__':
    db = load_db()
    #jobs = db.jobs_with(where='jobset83', state=SUCCESS)
    db_gan = load_db('/home/mcherti/dcgan/.lightjob')
    db_aa = load_db()
    jobs = []
    jobs += list(db_aa.jobs_with(state='success', where='jobset83'))
    jobs += list(db_aa.jobs_with(state='success', where='jobset86'))
    jobs += list(db_aa.jobs_with(state='success', where='jobset87'))
    jobs += list(db_aa.jobs_with(state='success', where='jobset88'))
    jobs += list(db_aa.jobs_with(state='success', where='jobset89'))
    jobs += list(db_aa.jobs_with(state='success', where='jobset90'))
    jobs_aa = to_generation(jobs)
    for j, j_gen in zip(jobs, jobs_aa):
        if j_gen and j:
            j_gen['content'] = j['content']
            j_gen['where'] = j['where']
    jobs_aa = list(filter(lambda j:j is not None, jobs_aa))
    jobs_gan = db_gan.jobs_with(state='success')
    for j in jobs_gan:
        j['content']['model_name'] = 'gan'
        j['where'] = ''
    jobs_gen = jobs_aa + jobs_gan
    hp = get_hypers(jobs_gen)
    hp.to_csv('hypers.csv')
