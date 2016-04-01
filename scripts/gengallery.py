import os
import json
from collections import defaultdict


def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


if __name__ == "__main__":
    from lightjob.db import DB, SUCCESS, RUNNING, AVAILABLE, ERROR
    from lightjob.cli import get_dotfolder
    import subprocess

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--folder', type=str, default='gallery')
    parser.add_argument('--nbpages', type=int, default=3, required=False)
    args = parser.parse_args()
    out_folder = args.folder
    nbpages = args.nbpages

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
        img_filename = os.path.join(folder, "iterations", "{:04d}.png".format(iteration))
        freq_filename = os.path.join(folder, "csv", "fixedpointshistogram_ylog.png")
        if not os.path.exists(img_filename):
            continue
        if not os.path.exists(freq_filename):
            continue
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
        cmd = "montage {} -tile x4 -geometry {}x{}+50+1 {}.png".format(cur_images, w, h, out)
        subprocess.call(cmd, shell=True)

        cur_freqs = freqs[first:last]

        w, h = 800, 800
        cur_images = ["\( {} -set label '{}' \)".format(img, caption) for img, caption in zip(cur_freqs, captions)]
        cur_images = " ".join(cur_images)
        out = os.path.join(out_folder, model_name, "freqs", "page{:04d}".format(pg))
        cmd = "montage {} -tile x4 -geometry {}x{} {}.png".format(cur_images, w, h, out)
        subprocess.call(cmd, shell=True)

    save_imgs(0, nb, pg=0)
    while first < nb:
        print("page {}".format(pg))
        last = first + per_page
        save_imgs(first, last)
        pg += 1
        first = last
