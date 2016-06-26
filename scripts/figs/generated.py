import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from lightjob.cli import load_db
from tqdm import tqdm
from skimage.io import imread

db = load_db()

J = db.jobs_with(state='success', type='generation')
for j in tqdm(J):
    id_ = j['summary']
    img_filename = 'jobs/results/{}/final1000.png'.format(id_)
    if not os.path.exists(img_filename):
        continue
    img = imread(img_filename)
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(img, interpolation='none', cmap='gray')
    plt.axis('off')
    plt.title('id={}'.format(id_))
    plt.savefig('figs/generated/{}.png'.format(id_))
    fig.tight_layout()
    plt.close(fig)
