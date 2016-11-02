import numpy as np
from scipy.stats import spearmanr
from lightjob.cli import load_db
import sys
from itertools import permutations, combinations
sys.path.append('.')
from tools.common import to_generation
import pandas as pd
from tabulate import tabulate

db = load_db()
jobs = db.jobs_with(state='success', where='jobset83')
jobs_gen = to_generation(jobs)
letterness = [
'diversity_count_letters_85',
'max_letters',
]
digitness = [
'diversity_count_digits_85',
'max_digits',
]
metrics = map(lambda m:'stats.out_of_the_box_classification.letterness.'+m, letterness)
metrics += map(lambda m:'stats.out_of_the_box_classification.letterness.'+m, digitness)
metrics = metrics + ['stats.out_of_the_box_classification.fonts.objectness']
metrics = metrics + ['stats.parzen_ll.ll_parzen_mean']
ordering = {}
def rename(m):
    if m == 'parzen_ll.ll_parzen_mean':
        return 'digit_parzen'
    return m
for m in metrics:
    indices = np.arange(len(jobs))
    scores = map(lambda j:db.get_value(j, m, if_not_found=np.nan), jobs_gen)
    scores = np.array(scores)
    indices = filter(lambda ind:not np.isnan(scores[ind]), indices)
    print(len(indices))
    indices = sorted(indices, key=lambda i:scores[i])
    indices = indices[::-1]
    ordering[m] = indices
rows = []
for m1, m2 in combinations(metrics, r=2):
    o1, o2 = ordering[m1], ordering[m2]
    rho, pvalue = spearmanr(o1, o2)
    m1 = '.'.join(m1.split('.')[-2:])
    m1 = rename(m1)
    m2 = '.'.join(m2.split('.')[-2:])
    m2 = rename(m2)
    rows.append({'m1': m1, 'm2': m2, 'rho': rho, 'pvalue': pvalue})
df = pd.DataFrame(rows)
df = df.sort_values(by='rho', ascending=False)
df.to_csv('exported_data/spearman.csv', index=False)
print(tabulate(df, headers='keys'))
