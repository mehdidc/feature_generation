import subprocess
import glob
import shutil

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb', default=1, type=int, required=False)
    parser.add_argument('-l', '--list', nargs='+', required=False)
    parser.add_argument('-f', '--force', type=bool, required=False)
    args = parser.parse_args()
    nb = args.nb
    force = args.force
    l = args.list

    if l is not None:
        jobs = l
    else:
        jobs = glob.glob("jobs/available/*")
        jobs = jobs[0:nb]
    print("Number of jobs to run : {}".format(nb))
    print(jobs)
    for j in jobs:
        with open(j) as fd:
            cmd = fd.read()
        cmd = "./launch {}".format(cmd)
        try:
            shutil.move(j, "jobs/running")
        except shutil.Error:
            if force:
                pass
            else:
                raise
        print(cmd)
        subprocess.call(cmd, shell=True)
