import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from lightjob.cli import load_db
from tqdm import tqdm
from skimage.io import imread
import json
from collections import OrderedDict
import argparse


def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


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

    new_J = []
    for j in J:
        id_ = j['summary']
        ref_id_ = j['content']['model_summary']
        jref = db.get_job_by_summary(ref_id_)
        if where and jref['where'] != where:
            continue
        new_J.append(j)
    print('Nb of jobs : {}'.format(len(new_J)))
    for j in tqdm(new_J):
        id_ = j['summary']
        ref_id_ = j['content']['model_summary']
        img_filename = 'jobs/results/{}/final1000.png'.format(id_)
        jref = db.get_job_by_summary(ref_id_)
        content = OrderedDict()
        content.update(jref['content'])
        content['id'] = id_
        jref_s = json.dumps(content, indent=4)
        if not os.path.exists(img_filename):
            print('{} does not exist, skip'.format(img_filename))
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
