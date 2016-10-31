import sys
import os
sys.path.append('/home/mcherti/work/code/feature_generation')
from lightjob.cli import load_db
import joblib
import numpy as np
from joblib import Parallel, delayed
import copy
import pandas as pd
from tools.common import to_generation
db = load_db()
jobs = db.jobs_with(state='success', type='training')
jobs_gen = to_generation(jobs)
for train, gen in zip(jobs, jobs_gen):
    if not gen:continue
    db.update({'models_generation_summary': gen['summary']}, train['summary'])
