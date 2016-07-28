import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from lightjob.cli import load_db
from tqdm import tqdm
from skimage.io import imread
import json
from collections import OrderedDict

import sys

def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


if len(sys.argv) == 2 and sys.argv[1] == 'per_jobset':
    per_jobset = True
else:
    per_jobset = False



db = load_db()

J = db.jobs_with(state='success', type='generation')
for j in tqdm(J):
    id_ = j['summary']
    ref_id_ = j['content']['model_summary']
    img_filename = 'jobs/results/{}/final1000.png'.format(id_)
    jref = db.get_job_by_summary(ref_id_)

    content = OrderedDict()
    content.update(jref['content'])
    content['id'] = id_
    jref_s = json.dumps(content, indent=4)
    if not os.path.exists(img_filename):
        continue
    img = imread(img_filename)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(img, interpolation='none', cmap='gray')
    plt.axis('off')
    plt.title(jref_s, fontsize=8)
    if per_jobset is False:
        plt.savefig('figs/generated/{}.png'.format(id_))
    else:
        where = jref['where']
        mkdir_path('figs/generated/{}'.format(where))
        plt.savefig('figs/generated/{}/{}.png'.format(where, id_))
    fig.tight_layout()
    plt.close(fig)
