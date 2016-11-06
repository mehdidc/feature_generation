import numpy as np
from scipy.stats import spearmanr
from lightjob.cli import load_db
import sys
from itertools import permutations, combinations
sys.path.append('.')
from tools.common import to_generation
import pandas as pd
from tabulate import tabulate
import sys

db_gan = load_db('/home/mcherti/dcgan/.lightjob')
db_aa = load_db()
jobs = db_aa.jobs_with(state='success', where='jobset83')
jobs_aa = to_generation(jobs)
jobs_gan = db_gan.jobs_with(state='success')
jobs_gen = jobs_aa + jobs_gan
db = db_aa
letterness = [
'diversity_count_letters_99',
'diversity_max_letters',
]
digitness = [
'diversity_count_digits_99',
'diversity_max_digits',
]
metrics = map(lambda m:'stats.out_of_the_box_classification.letterness.'+m, letterness)
metrics += map(lambda m:'stats.out_of_the_box_classification.letterness.'+m, digitness)
metrics += ['stats.out_of_the_box_classification.fonts.objectness']
metrics = metrics + ['stats.parzen_digits.mean', 'stats.parzen_letters.mean']
metrics = metrics + ['stats.out_of_the_box_classification.m2.objectness']

for m in metrics:
    print(m)
ordering = {}
summaries = {}

def m1_rename(x):
    x = x.replace('stats.', '')
    x = x.replace('.mean', '')
    x = x.replace('out_of_the_box_classification', '')
    x = x.replace('.letterness.', '')
    x = x.replace('.fonts', 'letters')
    x = x.replace('diversity_', '')
    x = x.replace('_99', '')
    x = x.replace('.m2', 'digits')
    return x

def m2_rename(x):
    x = rename(x)
    return x

nick = {
    'stats.out_of_the_box_classification.letterness.diversity_count_letters_99': 'lett_cnt',
    'stats.out_of_the_box_classification.letterness.diversity_max_letters': 'lett_max',
    'stats.out_of_the_box_classification.letterness.diversity_count_digits_99': 'dig_cnt',
    'stats.out_of_the_box_classification.letterness.diversity_max_digits': 'dig_max',
    'stats.out_of_the_box_classification.fonts.objectness': 'lett_obj',
    'stats.parzen_digits.mean': 'dig_parz',
    'stats.parzen_letters.mean': 'lett_parz',
    'stats.out_of_the_box_classification.m2.objectness': 'dig_obj'
}

def rename(m):
    return nick[m]

m1_cols = []
m2_cols = []
all_indices = set()
for m in metrics:
    indices = np.arange(len(jobs_gen))
    scores = map(lambda j:db.get_value(j, m, if_not_found=np.nan), jobs_gen)
    scores = np.array(scores)
    indices = filter(lambda ind:not np.isnan(scores[ind]), indices)

    summaries[m] = set(jobs_gen[ind]['summary'] for ind in indices)
    indices = sorted(indices, key=lambda i:scores[i])
    indices = indices[::-1]
    for ind in indices[0:100]:
        all_indices.add(ind)
    print(m, len(indices))

for m in metrics:
    indices = np.arange(len(jobs_gen))
    scores = map(lambda j:db.get_value(j, m, if_not_found=np.nan), jobs_gen)
    scores = np.array(scores)
    indices = filter(lambda ind:not np.isnan(scores[ind]), indices)

    summaries[m] = set(jobs_gen[ind]['summary'] for ind in indices)
    indices = sorted(indices, key=lambda i:scores[i])
    indices = indices[::-1]
    o = [ind for ind in indices if ind in all_indices]
    #o = indices
    r1 = m1_rename(m)
    r2 = m2_rename(m)
    
    ordering[r1] = o
    ordering[r2] = o
    
    m1_cols.append(r1)
    m2_cols.append(r2)

#print(m1_cols)
#print(m2_cols)
m1_cols = ['count_letters', 'max_letters', 'letters.objectness', 'parzen_letters', 'count_digits',  'max_digits',  'digits.objectness', 'parzen_digits', ]
m2_cols = ['lett_cnt', 'lett_max', 'lett_obj', 'lett_parz', 'dig_cnt', 'dig_max', 'dig_obj', 'dig_parz']
table = [['' for _ in range(len(metrics))] for _ in range(len(metrics))]
rhos = {}
for m1 in m1_cols:
    for m2 in m2_cols:
        o1, o2 = ordering[m1], ordering[m2]
        rho, pvalue = spearmanr(o1, o2)
        rhos[(m1, m2)] = rho

for i1, m1 in enumerate(m1_cols):
    for i2, m2 in enumerate(m2_cols):
        if i1 == i2:
            r = 1
        elif i1 > i2:
            r = 'x'
        else:
            r = '{:.2f}'.format(rhos[(m1, m2)])
        table[i1][i2] = r
df = pd.DataFrame(table)
df.index = m1_cols
df.columns = m2_cols
print(df)
print(tabulate(df, headers='keys', tablefmt='latex'))
