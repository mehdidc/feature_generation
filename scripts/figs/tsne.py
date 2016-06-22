import matplotlib as mpl
mpl.use('Agg')
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from lightjob.cli import load_db

db = load_db()

J = db.jobs_with(state='success', type='generation')
for j in J:
    id_ = j['summary']
    jref_s = j['content']['model_summary']
    jref = db.get_job_by_summary(jref_s)
    filename = 'jobs/results/{}/tsne_input.csv'.format(id_)
    if not os.path.exists(filename):
        continue
    img_filename = 'jobs/results/{}/final1000.png'.format(id_)
    if not os.path.exists(img_filename):
        continue
    img_content = imread(img_filename)
    data = pd.read_csv(filename)
    data = data.values
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 1], data[:, 2], marker='+')
    plt.subplot(1, 2, 2)
    plt.imshow(img_content, cmap='gray')
    plt.title(id_+'/'+jref_s)
    plt.savefig('figs/tsne/{}.png'.format(id_))
    plt.close(fig)
