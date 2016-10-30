import matplotlib as mpl
import os
from gengallery import gengallery

from genstats import genstats

if os.getenv("DISPLAY") is None:  # NOQA
    mpl.use('Agg')  # NOQA

import click
from lightjob.cli import load_db
from lightjob.db import SUCCESS


@click.group()
def main():
    pass

@click.command()
@click.option('--model', help='model', required=False, default=None)
@click.option('--where', help='where', required=True)
@click.option('--folder', help='folder', required=False, default='figs/gallery')
@click.option('--nbpages', help='nbpages', required=False, default=-1)
@click.option('--limit', help='limit', required=False, default=None)
@click.option('--show-freqs/--no-show-freqs', help='show_freqs', required=False, default=False)
@click.option('--force/--no-force', help='force', required=False, default=False)
def gallery(model, where, folder, nbpages, limit, show_freqs, force):
   jobs = load_jobs(model, where)
   limit = int(limit)
   gengallery(jobs,
              limit=limit,
              use_filtering=True,
              out_folder=folder,
              nbpages=nbpages, where=where,
              show_freqs=show_freqs,
              force=force)

@click.command()
@click.option('--model', help='model', required=False, default=None)
@click.option('--where', default='', help='where', required=False)
@click.option('--n_jobs', help='n_jobs', required=False, default=1)
@click.option('--stats', help='stats to compute (otherwise will compute everything) separeted by commas', required=False, default=None)
@click.option('--force/--no-force', help='force', required=False, default=False)
@click.option('--type', default='generation', help='type', required=False)
def stats(model, where, n_jobs, stats, force, type):
    if where == '':
        where = None
    jobs = load_jobs(model, where, type_=type)
    db = load_db()
    if stats is not None:
        stats = stats.split(',')
    genstats(jobs, db, n_jobs=n_jobs, force=force, filter_stats=stats)


def load_jobs(model_name, where, type_="generation"):
    db = load_db()
    jobs = []
    if type_ == 'generation':
        ref_jobs = set(map(lambda j:j['summary'], db.jobs_with(where=where)))
    for j in db.jobs_with(state=SUCCESS, type=type_):
        j = dict(j)
        if type_ == "generation":
            s = j['content']['model_summary']
            if where and s not in ref_jobs:
                continue
            ref_job = db.get_job_by_summary(j['content']['model_summary'])
            model_details = ref_job['content']
            j['ref_job'] = dict(ref_job)
            if where and ref_job['where'] != where:
                continue
        
        if model_name and model_details['model_name'] != model_name:
            continue
        jobs.append(j)
    return jobs

if __name__ == "__main__":
    main.add_command(stats)
    main.add_command(gallery)
    main()
