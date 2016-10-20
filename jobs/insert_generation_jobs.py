import os
import json

import click


def jobset_standard(jobs):
    for j in jobs:
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
            batch_size=1024
        )
        yield j, 'simple_genetic', params

def jobset_fast(jobs):
    # for iclr
    for j in jobs:
        params = {
            "batch_size": 256,
            "nb_samples": 1000,
            "nb_iter": 100,
            "do_sample": False,
            "do_binarize": True,
            "do_gaussian_noise": False,
            "do_noise": False,
            "thresh": "moving"
        }
        yield j, 'iterative_refinement', params

@click.command()
@click.option('--where', default='', help='jobset name', required=False)
@click.option('--jobset', default='jobset_standard', help='t', required=False)
@click.option('--nb', default=None, help='nb of repetitions', required=False)
@click.option('--budget', default=180, help='budget in min', required=False)
def insert(where, jobset, nb, budget):
    from lightjob.db import DB, SUCCESS
    from lightjob.cli import load_db
    from lightjob.utils import summarize
    db = load_db()
    kw = {}
    if where:
        kw['where'] = where
    jobs = db.jobs_with(state=SUCCESS, type="training", **kw)
    if nb:
        nb = int(nb)
        jobs = jobs[0:nb]
    fn = globals()[jobset]
    job_params = fn(jobs)
    print("Number of jobs : {}".format(len(jobs)))
    nb_insert = 0
    for job, what, params in job_params:
        sref = job['summary']
        d = {}
        d['model_summary'] = sref
        check = {}
        check['params'] = params
        check['dataset'] = job['content']['dataset']
        check['filename'] = 'jobs/results/{}/model.pkl'.format(sref)
        check['what'] = what
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
        print(cmd)
        print(params)
        nb_insert += db.safe_add_job(d, cmd=cmd, budget=budget, type="generation", where=jobset)

    print("Total number of jobs added : {}".format(nb_insert))

if __name__ == "__main__":
    insert()
