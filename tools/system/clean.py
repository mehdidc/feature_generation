
if __name__ == '__main__':
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS
    import shutil
    import os
    from joblib import dump
    import glob
    from tqdm import tqdm
    from skimage.io import imread
    import numpy as np
    db = load_db()
    def rm(path):
        if not os.path.exists(path):
            return
        print('Removing {}'.format(path))
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    for j in tqdm(db.jobs_with(state=SUCCESS, type="training")):
        j = dict(j)
        # training job
        folder = 'jobs/results/{}'.format(j['summary'])
        rm(os.path.join(folder, 'features'))
        rm(os.path.join(folder, 'recons'))
        rm(os.path.join(folder, 'out'))

    for j in tqdm(db.jobs_with(state=SUCCESS, type="generation")):
        j = dict(j)
        # generation job of the training job
        folder = 'jobs/results/{}'.format(j['summary'])
        rm(os.path.join(folder, 'iterations'))
        samples = glob.glob(os.path.join(folder, 'final/*.png'))
        samples = filter(lambda s:s.startswith('0'), samples)
        filename = folder + '/images.npz'
        if not os.path.exists(filename):
            samples = sorted(samples, key=lambda k:int(k.split('/')[-1].split('.')[0]))
            samples = map(imread, samples)
            samples = np.array(samples)
            dump(samples, filename, compress=9)
            rm(os.path.join(folder, 'final'))
