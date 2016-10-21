
if __name__ == '__main__':
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS, RUNNING, PENDING
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

    all_folders = set()
    folders = []
    for j in tqdm(db.jobs_with(type="training")):
        j = dict(j)
        folder = 'jobs/results/{}'.format(j['summary'])
        folders.append(folder)
        if j['state'] == RUNNING or j['state'] == PENDING:
            continue
        if j['state'] != SUCCESS:
            rm(os.path.join(folder))
            continue
        # training job
        rm(os.path.join(folder, 'features'))
        rm(os.path.join(folder, 'recons'))
        rm(os.path.join(folder, 'out'))
    
    for j in tqdm(db.jobs_with(type="generation")):
        j = dict(j)
        folder = 'jobs/results/{}'.format(j['summary'])
        folders.append(folder)
        if j['state'] == RUNNING or j['state'] == PENDING:
            continue
        if j['state'] != SUCCESS:
            rm(os.path.join(folder))
            continue
        # generation job of the training job
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
    all_folders |= set(folders)

    remain_folders = set(glob.glob('jobs/results/*')) 
    remain_folders -= all_folders
    for f in remain_folders:
        if not os.path.isdir(f):
            pass
        s = f.split('/')[-1]
        if s == 'iccc':
            print(s)
            continue
        if db.job_exists_by_summary(s):
            print(db.get_state_of(s))
        else:
            rm(f)
