import os
import json

import click

@click.command()
@click.option('--where', default='', help='jobset name', required=False)
@click.option('--nb', default=1, help='nb of repetitions', required=False)
@click.option('--mode', default='standard', help='standard/moving', required=False)
@click.option('--budget', default=180, help='budget in min', required=False)
def insert(where, nb, mode, budget):
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
    if mode == 'standard':
        up_binarize = 0.5
        batch_size = 1024
    elif mode == 'moving':
        up_binarize = 'moving'
        batch_size = 256
    else:
        raise Exception('bad mode')
    params = dict(
        op_params=[],
        op_names=[],
        nb_iterations=100,
        initial='random',
        initial_size=10000,
        layer_name='input',
        reconstruct=True,
        up_binarize=up_binarize,
        down_binarize=None,
        sort=True,
        tol=0,
        batch_size=batch_size
    )
    for job in jobs:
        sref = job['summary']
        d = {}
        d['model_summary'] = sref
        check = {}
        check['params'] = params
        check['dataset'] = job['content']['dataset']
        check['filename'] = 'jobs/results/{}/model.pkl'.format(sref)
        check['what'] = 'simple_genetic'
        d['check'] = check
        s = summarize(d)
        cmd = "sbatch --time={time} --output={out} --error={out} {launch} invoke check --update-db=1 --what={what} --dataset={dataset} --filename={filename} --params={params} --folder={folder}"
        cmd =  cmd.format(
            time=budget,
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
