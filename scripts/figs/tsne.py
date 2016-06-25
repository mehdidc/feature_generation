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
    filename = 'jobs/results/{}/tsne_latent.csv'.format(id_)
    if not os.path.exists(filename):
        continue
    img_filename = 'jobs/results/{}/final1000.png'.format(id_)
    if not os.path.exists(img_filename):
        continue
    img_content = imread(img_filename)
    data = pd.read_csv(filename)

    for c in ('x', 'y'):
        data[c] = (data[c] - data[c].mean()) / data[c].std()

    fig = plt.figure(figsize=(20, 30))
    plt.subplot(2, 1, 1)
    plt.scatter(data["x"], data["y"], marker='+', cmap='gray')
    plt.title(id_+'/'+jref_s)
    plt.subplot(2, 1, 2)
    plt.imshow(img_content, cmap='gray', interpolation='none')
    plt.title(id_+'/'+jref_s)
    plt.savefig('figs/tsne/{}.png'.format(id_))
    plt.show()
    plt.close(fig)
