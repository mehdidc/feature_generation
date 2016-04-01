import subprocess

if __name__ == "__main__":
    import argparse
    from lightjob.db import DB, AVAILABLE, RUNNING
    from lightjob.cli import get_dotfolder

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb', default=1, type=int, required=False)
    parser.add_argument('-l', '--list', nargs='+', required=False)
    parser.add_argument('-f', '--force', type=bool, required=False)
    parser.add_argument('-w', '--where', default=None, type=str, required=False)
    parser.add_argument('-t', '--type', default=None, type=str, required=False)
    args = parser.parse_args()
    nb = args.nb
    force = args.force
    l = args.list
    where = args.where
    type_ = args.type

    folder = get_dotfolder()
    db = DB()
    db.load(folder)

    if l is not None:
        jobs = []
        for s in l:
            j = db.get_job_by_summary(s)
            if j:
                jobs.append(j)
    else:
        extra = dict()
        if where is not None:
            extra["where"] = where
        if type_ is not None:
            extra["type"] = type_
        jobs = db.jobs_with(state=AVAILABLE, **extra)
    jobs = jobs[0:nb]
    nb = len(jobs)
    print("Number of jobs to run : {}".format(nb))
    for j in jobs:
        cmd = j["cmd"]
        print(cmd)
        db.modify_state_of(j['summary'], RUNNING)
        subprocess.call(cmd, shell=True)
