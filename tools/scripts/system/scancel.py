import sys
from lightjob.cli import load_db
from lightjob.db import RUNNING
import subprocess

db = load_db()
values = db.get_values('slurm_job_id', state=RUNNING)
values = list(values)
values = [v['slurm_job_id'] for v in values]
for v in values:
    cmd = 'scancel {}'.format(v)
    subprocess.call(cmd, shell=True)
