import glob
import json
import os
import subprocess
import sys
import re

from collections import Counter, OrderedDict, defaultdict
from joblib import Parallel, delayed

import numpy as np
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


def get_normalized_counts(hash_matrix):
    hm = hash_matrix
    cnt = Counter(hm)
    s = sum(cnt.values())
    for k, v in cnt.items():
        cnt[k] = float(cnt[k]) / s
    return cnt


def get_hash_to_int_map(hash_matrix):
    K = {}
    for i, h in enumerate(hash_matrix):
        if h not in K:
            K[h] = i
    return K

def call(cmd, shell=True):
    logger.info(cmd)
    subprocess.call(cmd, shell=shell)

def gengallery(jobs, limit=None, use_filtering=True, out_folder='gallery', nbpages=-1, where='jobset1', show_freqs=True, force=False):
    images = []
    plots = []
    captions = []
    commands = []
    for j in jobs:
        ref_job = j['ref_job']
        model_details = ref_job['content']
        folder = "jobs/results/{}".format(j['summary'])
        hash_matrix_filename = os.path.join(folder, "csv", "hashmatrix.npy")

        if limit is None:
            # if not limit consider all images
            iterations = map(lambda name: int(name.split(".")[0]),
            os.listdir(os.path.join(folder, "iterations")))
            iteration = max(iterations)
            img_filename = os.path.join(
                folder, "iterations", "{:04d}.png".format(iteration))
        else:
            # if limit sort them by frequency and take only the first ones
            hash_matrix = np.load(hash_matrix_filename)
            hm = hash_matrix
            cnt = get_normalized_counts(hash_matrix)
            if use_filtering:
                K = get_hash_to_int_map(hash_matrix)
                indices = K.values()
                indices = indices[0:limit]
                indices = sorted(indices, key=lambda ind: -cnt[hm[ind]])
            else:
                indices = range(0, limit)
            filenames = glob.glob(os.path.join(folder, 'final', '*.png'))
            filenames = sorted(filenames)
            filenames = [filenames[ind] for ind in indices]
            texts = ["{:.2f}".format(100. * cnt[hm[ind]]) for ind in indices]
            if show_freqs:
                filenames = ["\( {} -set label '{}' \)".format(img, txt)
                             for img, txt in zip(filenames, texts)]
                border = "-geometry +4+4"
            else:
                border = "-geometry +0+0"

            filenames = " ".join(filenames)

            img_filename = os.path.join(folder, "final{}.png".format(limit))
            if not os.path.exists(img_filename) or force:
                cmd = "montage {} -pointsize 8.5 {} {}"
                cmd = cmd.format(filenames, border, img_filename)
                commands.append(cmd)

        plot_dict = {}
        plots.append(plot_dict)
        #if not os.path.exists(img_filename):
        #    continue
        images.append(img_filename)
        c = OrderedDict()
        c.update(model_details)
        c['id'] = j['summary']
        if "stats" in j:
            c["stats"] = j["stats"]
        captions.append(c)

    #for cmd in commands:
    #    logger.info('generating {}'.format(img_filename))
    #    subprocess.call(cmd, shell=True)
    Parallel(n_jobs=-1, backend='threading')(delayed(call)(cmd, shell=True) for cmd in commands)
    logger.info('Total of {}'.format(len(images)))
    # filter indices
    indices = range(len(images))
    indices = sorted(indices, key=lambda i: captions[i].items())
    images = [images[i] for i in indices]
    captions = [captions[i] for i in indices]
    plots = [plots[i] for i in indices]

    p_vals = defaultdict(set)
    p_model_vals = defaultdict(set)
    for c in captions:
        for k, v in c.items():
            if type(v) == dict:
                continue
            p_vals[k].add(v)
        for k, v in c['model_params'].items():
            if type(v) == list:
                v = tuple(v)
            p_model_vals[k].add(v)
    for k in p_vals.keys():
        if len(p_vals[k]) == 1:
            del p_vals[k]
    for k in p_model_vals.keys():
        if len(p_model_vals[k]) == 1:
            del p_model_vals[k]
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
            if 'stats' in c:
                cn['stats'] = c['stats']
            cn['id'] = c['id']
        captions[i] = json.dumps(cn, indent=4)
    if nbpages == -1:
        per_page = 1
    else:
        per_page = len(images) / nbpages
    first = 0
    pg = 1

    print(where)
    prefix, nb, _ = re.split('(\d+)', where)
    nb = int(nb)
    where_nicer = '{}{:03d}'.format(prefix, nb)

    mkdir_path(os.path.join(out_folder, where_nicer, "generated"))
    plot_names = plots[0].keys()
    for a in plot_names:
        mkdir_path(os.path.join(out_folder, where_nicer, a))
    nb = len(images)

    def save_imgs(first, last, pg=0, w=1500, h=1500, wp=800, hp=800):
        cur_images = images[first:last]
        cur_captions = captions[first:last]
        cur_images = ["\( {} -set label '{}' \)".format(img, caption) 
                      for img, caption in zip(cur_images, cur_captions)]
        cur_images = " ".join(cur_images)
        out = os.path.join(
            out_folder, where_nicer,
            "generated",
            "page{:04d}".format(pg))
        if w is not None and h is not None:
            sz = '{}x{}'.format(w, h)
        else:
            sz = ''
        # TODO if nb items is 1 just copy the image
        cmd = "montage {} -geometry {}+50+1 {}.png".format(cur_images, sz, out)
        logger.info(cmd)
        subprocess.call(cmd, shell=True)

        cur_plots = plots[first:last]

        for p in plot_names:
            cur_images = ["\( {} -set label '{}' \)".format(img[p], caption)
                          for img, caption in zip(cur_plots, cur_captions)]
            cur_images = " ".join(cur_images)
            out = os.path.join(out_folder, where_nicer, p, "page{:04d}".format(pg))
            if wp is not None and hp is not None:
                sz = '-geometry {}x{}'.format(wp, hp)
            else:
                sz = ''
            cmd = "montage {} {}.png".format(cur_images, sz, out)
            logger.info(cmd)
            subprocess.call(cmd, shell=True)

    while first < nb:
        print("page {}".format(pg))
        last = first + per_page
        save_imgs(first, last, pg=pg, w=None, h=None, wp=None, hp=None)
        pg += 1
        first = last
