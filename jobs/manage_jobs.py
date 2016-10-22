from lightjob.cli import load_db
import click
import json
import pandas as pd
import gzip
import os
#source : https://github.com/pallets/click/issues/108
#pass_state = click.make_pass_decorator(State, ensure=True)

@click.group()
def main():
    pass

def common(f):
    f = click.option('--where', default=None, help='where', required=False)(f)
    f = click.option('--type', default=None, help='where', required=False)(f)
    f = click.option('--ref_where', default=None, help='ref_where', required=False)(f)
    f = click.option('--state', default=None, help='state', required=False)(f)
    f = click.option('--details', default=False, help='show details', required=False)(f)
    f = click.option('--finished-after', default='', help='came after', required=False)(f)
    return f

@click.command()
@common
def show(**kwargs):
    jobs = filter_jobs(**kwargs)
    finished_after = kwargs['finished_after']
    details = kwargs['details']
    if finished_after != '':
        ref_date = pd.to_datetime(finished_after)
    else:
        ref_date = None
    for j in jobs:
        j = dict(j)
        if ref_date:
            if not ('life' in j and j['life']):
                continue
            dt = pd.to_datetime(j['life'][-1]['dt'])
            if(dt < ref_date):
                continue
        if details:
            print(json.dumps(j, indent=4))
        else:
            print(j['summary'])

@click.command()
@common
@click.option('--dontcare', help='', default=None, required=False)
def delete(**kwargs):
    dontcare = kwargs['dontcare']
    jobs = filter_jobs(*args, **kwargs)
    if dontcare is None:
        print('Please specify that you dont care if you want to delete...')
        return
    if dontcare == False:
        print('ok you care, so I will not delete anything...')
        return
    for j in jobs:
        print('Deleting {}'.format(j['summary']))
        db.delete(dict(summary=j['summary']))

@click.command()
@common
def detect_err(*args, **kwargs):
    jobs = filter_jobs(*args, **kwargs)
    for j in jobs:
        s = j['summary']
        if has_exception(s):
            print(s)

def filter_jobs(where=None, type=None, ref_where=None, state=None, dontcare=None, details=None, finished_after=None):
    db = load_db()
    kw = {}
    if where:
        kw['where'] = where
    if type:
        kw['type'] = type
    if state:
        kw['state'] = state
    if ref_where:
        ref_jobs = set(map(lambda j:j['summary'], db.jobs_with(where=ref_where)))
    else:
        ref_jobs = set()
    def accepted(job):
        if ref_where :
            if 'model_summary' not in job['content']:
                return False
            s = job['content']['model_summary']
            return s in ref_jobs
        return True
    jobs = db.jobs_with(**kw)
    jobs = filter(accepted, jobs)
    def get_key(j):
        if 'life' in j and j['life']:
            l = j['life'][-1]['dt']
            l = pd.to_datetime(l, infer_datetime_format=True)
            return l
        else:
            return pd.to_datetime('Fri Jan 01 00:00:40 1970')
    jobs = sorted(jobs, key=get_key)
    return jobs

def has_exception(s):
    filename = 'jobs/outputs/{}'.format(s)
    if os.path.exists(filename):
        data = open(filename).read()
    elif os.path.exists(filename + '.gz'):
        data = gzip.GzipFile(filename + '.gz').read()
    else:
        return False
    if "Traceback (most recent call last):" in data:
        return True
    if "DUE TO TIME LIMIT ***" in data:
        return True
    return False

if __name__ == '__main__':
    main.add_command(show)
    main.add_command(delete)
    main.add_command(detect_err)
    main()
