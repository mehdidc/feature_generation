import matplotlib as mpl
mpl.use('Agg')
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from lightjob.cli import load_db
import shutil

db = load_db()

J = db.jobs_with(state='success', type='generation')
for j in J:
    id_ = j['summary']
    img_filename = 'jobs/results/{}/final1000.png'.format(id_)
    if not os.path.exists(img_filename):
        continue
    shutil.copyfile(img_filename, 'figs/generated/{}.png'.format(id_))
