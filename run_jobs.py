import subprocess
import glob
import shutil

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb', default=1)
    args = parser.parse_args()
    nb = args.nb
    jobs = glob.glob("jobs/available/*")
    jobs = jobs[0:nb]
    print("Number of jobs to run : {}".format(nb))
    for j in jobs:
        with open(j) as fd:
            cmd = fd.read()
        cmd = "./launch {}".format(cmd)
        shutil.move(j, "jobs/running")
        subprocess.call(cmd, shell=True)
