import subprocess
import time
from lightjob.cli import load_db
from lightjob.db import PENDING, AVAILABLE
import click

@click.command()
@click.option('--nb', default=1, required=False)
@click.option('--where', default=None, required=False)
@click.option('--ref_where', default=None, required=False)
@click.option('--ref_where', default=None, required=False)
@click.option('--type', default=None, required=False)
@click.option('--sequential/--parallel', default=False, required=False)
def run(nb, where, ref_where, type, sequential):
    db = load_db()
    extra = dict()
    if where is not None:
        extra["where"] = where
    if type is not None:
        extra["type"] = type
    jobs = db.jobs_with(state=AVAILABLE, **extra)
    print("Number of jobs to run : {}".format(nb))
    if sequential:
        for j in jobs:
            db.modify_state_of(j['summary'], PENDING)
    for j in jobs:
        if nb == 0:
            break
        if ref_where is not None and 'model_summary' in j['content']:
            jref = db.get_job_by_summary(j['content']['model_summary'])
            if not jref or jref['where'] != ref_where:
                continue
            else:
                print('ok not skipping')
        cmd = j["cmd"]
        print(cmd)
        db.modify_state_of(j['summary'], PENDING)
        if sequential:
            cmd = cmd[cmd.find('invoke'):]
        subprocess.call(cmd, shell=True)
        time.sleep(0.5)
        nb -= 1

if __name__ == "__main__":
    run()
