import subprocess
import time
from lightjob.cli import load_db
from lightjob.db import PENDING, AVAILABLE, SUCCESS
import click
import sys
sys.path.append('.')
from tools.common import to_generation

@click.command()
@click.option('--nb', default=1, required=False)
@click.option('--where', default=None, required=False)
@click.option('--type', default=None, required=False)
@click.option('--sequential/--parallel', default=False, required=False)
def run(nb, where, type, sequential):
    db = load_db()
    extra = dict()
    if where is not None:
        extra["where"] = where
    if type == 'training':
        jobs = db.jobs_with(state=AVAILABLE, **extra)
    elif type == 'generation':
        jobs = db.jobs_with(state=SUCCESS, **extra)
        jobs = to_generation(jobs, state=AVAILABLE, db=db)
        jobs = filter(lambda j:j, jobs)
        jobs = jobs[0:nb]
    print("Number of jobs to run : {}".format(len(jobs)))
    if sequential:
        for j in jobs:
            db.modify_state_of(j['summary'], PENDING)
    for j in jobs:
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
