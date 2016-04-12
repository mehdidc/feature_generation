import os
import json
from collections import defaultdict
import os
import pandas as pd
import matplotlib as mpl
import glob
if os.getenv("DISPLAY") is None:  # NOQA
    mpl.use('Agg')  # NOQA


def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)

def loglogplot(hash_matrix, filename):
    from collections import Counter
    import matplotlib.pyplot as plt
    hm = hash_matrix
    cnt = Counter(hm)
    V = sorted(cnt.values(), reverse=True)
    V = np.array(V)
    fig = plt.figure(figsize=(15, 15))
    plt.bar(np.arange(len(V)), V)
    plt.ylabel("frequency")
    plt.xlabel("fixed point")
    plt.legend()
    plt.xlim((0, len(cnt)))
    plt.title("Frequency of fixed points")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(filename)
    plt.close(fig)


def get_powerlaw_exponent(hash_matrix):
    from collections import Counter
    import powerlaw
    hm = hash_matrix
    cnt = Counter(hm)

    s = np.argsort(cnt.values())[::-1]

    K = cnt.keys()
    K = [K[s[i]] for i in range(len(K))]
    K_to_int = {k: i + 1 for i, k in enumerate(K)}
    x = [K_to_int[v] for v in hm]
    results = powerlaw.Fit(x)
    return results.alpha, results.xmin, results.pdf()

if __name__ == "__main__":
    from lightjob.db import DB, SUCCESS, RUNNING, AVAILABLE, ERROR
    from lightjob.cli import get_dotfolder
    import subprocess
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--folder', type=str, default='gallery')
    parser.add_argument('--nbpages', type=int, default=1, required=False)
    parser.add_argument('--limit', type=int, default=None, required=False)
    args = parser.parse_args()
    out_folder = args.folder
    nbpages = args.nbpages
    limit = args.limit
    folder = get_dotfolder()
    db = DB()
    db.load(folder)

    images = []
    freqs = []
    captions = []
    model_name = args.model
    for j in db.jobs_with(state=SUCCESS, type="generation"):
        s = j['content']['model_summary']
        ref_job = db.get_job_by_summary(s)
        model_details = ref_job['content']
        if model_details['model_name'] != model_name:
            continue
        folder = "jobs/results/{}".format(j['summary'])
        iterations = map(lambda name: int(name.split(".")[0]),
                         os.listdir(os.path.join(folder, "iterations")))
        iteration = max(iterations)
        if limit is None:
            img_filename = os.path.join(folder, "iterations", "{:04d}.png".format(iteration))
        else:
            filenames = glob.glob(os.path.join(folder, 'final', '*.png'))
            filenames = sorted(filenames)
            filenames = filenames[0:limit]
            filenames = " ".join(filenames)

            img_filename = os.path.join(folder, "final{}.png".format(limit))
            if not os.path.exists(img_filename):
                cmd = "montage {} -geometry +2+2 {}".format(filenames, img_filename)
                subprocess.call(cmd, shell=True)

        freq_filename = os.path.join(folder, "csv", "fixedpointshistogram_xlog_ylog.png")
        if not os.path.exists(img_filename):
            continue
        if not os.path.exists(freq_filename):
            hash_matrix_filename = os.path.join(folder, "csv", "hashmatrix.npy")
            print(hash_matrix_filename)
            hash_matrix = np.load(hash_matrix_filename)
            alpha, xmin, pdf = get_powerlaw_exponent(hash_matrix)
            loglogplot(hash_matrix, freq_filename)
        freqs.append(freq_filename)
        images.append(img_filename)
        #c = json.dumps(model_details, indent=4)
        #c = ["{}={}".format(k, v) for k, v in model_details.items()]
        #c = " ".join(c)
        #c = c.replace("_", "-")
        #c = r"%s" % (c,)
        print(img_filename)
        captions.append(model_details)
    print(len(images))
    p_vals = defaultdict(set)
    p_model_vals = defaultdict(set)
    for c in captions:
        for k, v in c.items():
            if type(v) == dict:
                continue
            p_vals[k].add(v)
        for k, v in c['model_params'].items():
            p_model_vals[k].add(v)
    for k in p_vals.keys():
        if len(p_vals[k]) == 1:
            del p_vals[k]
            print(k)
    for k in p_model_vals.keys():
        if len(p_model_vals[k]) == 1:
            del p_model_vals[k]
            print(k)
    for i, c in enumerate(captions):
        if i == 0:
            cn = c
        else:
            cn = {}
            cn['model_params'] = {}
            for k in p_vals.keys():
                cn[k] = c[k]
            for k in p_model_vals.keys():
                cn['model_params'][k] = c['model_params'][k]
        captions[i] = json.dumps(cn, indent=4)

    per_page = len(images) / nbpages
    first = 0
    pg = 1
    mkdir_path(os.path.join(out_folder, model_name, "generated"))
    mkdir_path(os.path.join(out_folder, model_name, "freqs"))
    nb = len(images)

    def save_imgs(first, last, pg=0):
        w, h = 1500, 1500
        cur_images = images[first:last]
        cur_images = ["\( {} -set label '{}' \)".format(img, caption) for img, caption in zip(cur_images, captions)]
        cur_images = " ".join(cur_images)
        out = os.path.join(out_folder, model_name, "generated", "page{:04d}".format(pg))
        cmd = "montage {} -tile 4x -geometry {}x{}+50+1 {}.png".format(cur_images, w, h, out)
        print(cmd)
        subprocess.call(cmd, shell=True)

        cur_freqs = freqs[first:last]

        w, h = 800, 800
        cur_images = ["\( {} -set label '{}' \)".format(img, caption) for img, caption in zip(cur_freqs, captions)]
        cur_images = " ".join(cur_images)
        out = os.path.join(out_folder, model_name, "freqs", "page{:04d}".format(pg))
        cmd = "montage {} -tile 4x -geometry {}x{} {}.png".format(cur_images, w, h, out)
        print(cmd)
        subprocess.call(cmd, shell=True)

    #save_imgs(0, nb, pg=0)
    while first < nb:
        print("page {}".format(pg))
        last = first + per_page
        save_imgs(first, last, pg=pg)
        pg += 1
        first = last
