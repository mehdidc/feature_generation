import os
import json

if __name__ == "__main__":
    from lightjob.db import DB, SUCCESS
    from lightjob.cli import load_db
    from lightjob.utils import summarize
    db = load_db()
    jobs = db.jobs_with(state=SUCCESS, type="training")
    print("Number of jobs : {}".format(len(jobs)))
    nb = 0

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
        #print(cmd)
        folder = 'jobs/generations/{}'.format(s)
        nb += db.safe_add_job(d, cmd=cmd, type="generation")
        #if db.job_exists(d) and db.get_job_by_summary(s)['state'] != SUCCESS:
        #    print(db.get_job_by_summary(s)['cmd'])
        ##    db.job_update(s, dict(cmd=cmd))

    print("Total number of jobs added : {}".format(nb))
