import matplotlib as mpl
import os
from gengallery import gengallery

from genstats import genstats

if os.getenv("DISPLAY") is None:  # NOQA
    mpl.use('Agg')  # NOQA

if __name__ == "__main__":
    from lightjob.db import DB, SUCCESS
    from lightjob.cli import get_dotfolder
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default=None)
    parser.add_argument('--where', type=str, required=True)
    parser.add_argument('--folder', type=str, default='gallery')
    parser.add_argument('--nbpages', type=int, default=1, required=False,
                        help='-1 to use one page per  model')
    parser.add_argument('--limit', type=int, default=None, required=False)
    parser.add_argument('--njobs', type=int, default=-1, required=False)
    parser.add_argument('--show_freqs', default=False, action='store_true', required=False)
    parser.add_argument('--force', default=False, action='store_true', required=False)

    parser.add_argument('action', type=str, default='gallery')


    args = parser.parse_args()
    action = args.action
    out_folder = args.folder
    nbpages = args.nbpages
    limit = args.limit
    force = args.force
    where = args.where
    n_jobs = args.njobs
    show_freqs = args.show_freqs
    folder = get_dotfolder()
    db = DB()
    db.load(folder)

    images = []
    plots = []
    captions = []
    model_name = args.model

    jobs = []
    for j in db.jobs_with(state=SUCCESS, type="generation"):
        j = dict(j)
        s = j['content']['model_summary']
        ref_job = db.get_job_by_summary(s)
        model_details = ref_job['content']
        j['ref_job'] = dict(ref_job)
        if ref_job['where'] != where:
            continue
        if model_name and model_details['model_name'] != model_name:
            continue
        jobs.append(j)

    if action == 'gallery':
        gengallery(jobs,
                   limit=limit,
                   use_filtering=True,
                   out_folder=out_folder,
                   nbpages=nbpages, where=where,
                   show_freqs=show_freqs,
                   force=force)
    elif action == 'computestats':
        genstats(jobs, db, n_jobs=n_jobs)
