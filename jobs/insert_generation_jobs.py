import os
import json

import click

@click.command()
@click.option('--where', default='', help='jobset name', required=False)
@click.option('--nb', default=1, help='nb of repetitions', required=False)
def insert(where, nb):
    from lightjob.db import DB, SUCCESS
    from lightjob.cli import load_db
    from lightjob.utils import summarize
    db = load_db()
    kw = {}
    if where:
        kw['where'] = where
    jobs = db.jobs_with(state=SUCCESS, type="training", **kw)
    print("Number of jobs : {}".format(len(jobs)))
    nb = 0
    for up_binarize in (0.5,):
        params = dict(
            op_params=[],
            op_names=[],
            nb_iterations=100,
            initial='random',
            initial_size=10000,
            layer_name='input',
            reconstruct=True,
            up_binarize=0.5,
            down_binarize=None,
            sort=True,
            tol=0,
        )
        for job in jobs:
            sref = job['summary']
            d = {}
            d['model_summary'] = sref
            check = {}
            check['params'] = params
            print(job['content'].keys())
            check['dataset'] = job['content']['dataset']
            check['filename'] = 'jobs/results/{}/model.pkl'.format(sref)
            check['what'] = 'simple_genetic'
            d['check'] = check
            s = summarize(d)
            cmd = "sbatch --time={time} --output={out} --error={out} {launch} invoke check --update-db=1 --what={what} --dataset={dataset} --filename={filename} --params={params} --folder={folder}"
            cmd =  cmd.format(
                time=180,
                launch='scripts/launch_gpu',
                out="jobs/outputs/{}".format(s),
                what=check['what'],
                dataset=check['dataset'],
                filename=check['filename'],
                params=s,
                folder="jobs/results/{}".format(s)
            )
            folder = 'jobs/generations/{}'.format(s)
            nb += db.safe_add_job(d, cmd=cmd, type="generation")

    print("Total number of jobs added : {}".format(nb))

if __name__ == "__main__":
    insert()
