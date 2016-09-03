
if __name__ == '__main__':
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS
    import shutil
    import os
    db = load_db()
    def rm(path):
        if not os.path.exists(path):
            return
        print('Removing {}'.format(path))
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    for j in db.jobs_with(state=SUCCESS, type="training"):
        j = dict(j)
        # training job
        folder = 'jobs/results/{}'.format(j['summary'])
        rm(os.path.join(folder, 'features'))
        rm(os.path.join(folder, 'recons'))
        rm(os.path.join(folder, 'out'))

    for j in db.jobs_with(state=SUCCESS, type="generation"):
        j = dict(j)
        # generation job of the training job
        folder = 'jobs/results/{}'.format(j['summary'])
        rm(os.path.join(folder, 'iterations'))


