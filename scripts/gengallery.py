import os
import json
from collections import defaultdict, OrderedDict
import os
import pandas as pd
import matplotlib as mpl
import glob
from collections import Counter
from itertools import product
from numba import jit

if os.getenv("DISPLAY") is None:  # NOQA
    mpl.use('Agg')  # NOQA

def neighcorr(im, corrdata=None, pad=3):
    assert corrdata is not None
    i = 0
    for x in range(pad, im.shape[0] - pad):
        for y in range(pad, im.shape[1] - pad):
            pxc = im[x, y]
            ctot = 0
            for dx, dy in product((0, 1, -1), (0, 1, -1)):
                px = im[x+dx, y+dy]
                c = px * pxc
                corrdata[i] += c
            i += 1

def neighbcorr_filenames(filenames):
    from skimage.io import imread
    corr = []
    corrdata = defaultdict(int)
    for i, f in enumerate(filenames):
        if i % 100 == 0:
            print(i)
        im = imread(f)
        im = 2 * (im / im.max()) - 1
        assert set(im.flatten().tolist()) <= set([1, -1]), set(im.flatten().tolist())
        neighcorr(im, corrdata=corrdata, pad=3)
    nc = np.abs(np.array(corrdata.values())).mean() / len(filenames)
    return nc

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

def hash_matrix_to_int(hm):
    from collections import Counter
    cnt = Counter(hm)
    s = np.argsort(cnt.values())[::-1]
    K = cnt.keys()
    K = [K[s[i]] for i in range(len(K))]
    K_to_int = {k: i + 1 for i, k in enumerate(K)}
    x = [K_to_int[v] for v in hm]
    x = np.array(x)
    return x

def powerlawplot(hash_matrix, folder, force=False):
    from collections import Counter
    import powerlaw
    import matplotlib.pyplot as plt

    cat = {}
    filenamenoxmin = os.path.join(folder, "powerlawnoxmin.png")
    cat["powerlawnoxmin"] = filenamenoxmin

    filenamexminfull = os.path.join(folder, "powerlawxmin_full.png")
    cat["powerlawxminfull"] = filenamexminfull

    filenamexminwindow = os.path.join(folder, "powerlawxmin_window.png")
    cat["powerlawxminwindow"] = filenamexminwindow

    hm = hash_matrix
    cnt = Counter(hm)
    s = np.argsort(cnt.values())[::-1]
    K = cnt.keys()
    K = [K[s[i]] for i in range(len(K))]
    K_to_int = {k: i + 1 for i, k in enumerate(K)}
    x = [K_to_int[v] for v in hm]
    x = np.array(x)
    if not os.path.exists(filenamenoxmin) or force:
        fit = powerlaw.Fit(x, discrete=True, xmin=x.min())
        plt.clf()
        fig = plt.figure()
        try:
            fig2 = fit.plot_pdf(original_data=True, color='b', linewidth=2, label='original pdf')
            fit.power_law.plot_pdf(color='b', linestyle='--',
                                ax=fig2,
                                label=r"fit pdf ($\alpha={:.2f},\sigma={:.2f}$)".format(fit.alpha, fit.sigma))
            plt.axvline(fit.xmin, color='g', linestyle='--', label='xmin={}'.format(int(fit.xmin)))
            plt.xlabel('x')
            plt.ylabel('$p(x)$')
            plt.legend()
        except Exception as e:
            print(str(e))
        plt.savefig(filenamenoxmin)
        print(filenamenoxmin)
        plt.close(fig)

    if not os.path.exists(filenamexminfull) or not os.path.exists(filenamexminwindow) or force:
        fit = powerlaw.Fit(x, discrete=True)
        for o in (True, False):
            plt.clf()
            fig = plt.figure()
            try:
                fig2 = fit.plot_pdf(original_data=o, color='b', linewidth=2, label='original pdf')
                fit.power_law.plot_pdf(color='b', linestyle='--',
                                       ax=fig2,
                                       label=r"fit pdf ($\alpha={:.2f},\sigma={:.2f}$)".format(fit.alpha, fit.sigma))
                plt.axvline(fit.xmin, color='g', linestyle='--', label='xmin={}'.format(int(fit.xmin)))
                plt.xlabel('x')
                plt.ylabel('$p(x)$')
                plt.legend()
            except Exception as e:
                print(str(e))
            if o == True:
                plt.savefig(filenamexminfull)
                print(filenamexminfull)
            else:
                plt.savefig(filenamexminwindow)
                print(filenamexminwindow)
            plt.close(fig)
    return cat

if __name__ == "__main__":
    from lightjob.db import DB, SUCCESS, RUNNING, AVAILABLE, ERROR
    from lightjob.cli import get_dotfolder
    import subprocess
    import numpy as np
    import argparse
    from scipy.stats import skew
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default=None)
    parser.add_argument('--where', type=str, required=True)
    parser.add_argument('--folder', type=str, default='gallery')
    parser.add_argument('--nbpages', type=int, default=1, required=False, help='-1 to use one page per  model')
    parser.add_argument('--limit', type=int, default=None, required=False)
    use_filtering = True
    args = parser.parse_args()
    out_folder = args.folder
    nbpages = args.nbpages
    limit = args.limit
    where = args.where
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
        if ref_job['where'] != where:
            continue
        if model_name and model_details['model_name'] != model_name:
            continue
        print(model_details)
        folder = "jobs/results/{}".format(j['summary'])
        iterations = map(lambda name: int(name.split(".")[0]),
                         os.listdir(os.path.join(folder, "iterations")))
        iteration = max(iterations)
        if limit is None:
            img_filename = os.path.join(folder, "iterations", "{:04d}.png".format(iteration))
        else:
            hash_matrix_filename = os.path.join(folder, "csv", "hashmatrix.npy")
            hash_matrix = np.load(hash_matrix_filename)
            hm = hash_matrix
            cnt = Counter(hm)
            s = sum(cnt.values())
            for k, v in cnt.items():
                cnt[k] = float(cnt[k]) / s

            if use_filtering:
                K = {}
                for i, h in enumerate(hm):
                    if h not in K:
                        K[h] = i
                indices = K.values()
                indices = indices[0:limit]
                indices = sorted(indices, key=lambda ind:-cnt[hm[ind]])
            else:
                indices = range(0, limit)
            filenames = glob.glob(os.path.join(folder, 'final', '*.png'))
            filenames = sorted(filenames)
            filenames = [filenames[ind] for ind in indices]
            texts = ["{:.2f}".format(100. * cnt[hm[ind]]) for ind in indices]
            filenames = ["\( {} -set label '{}' \)".format(img, txt) for img, txt in zip(filenames, texts)]
            filenames = " ".join(filenames)

            img_filename = os.path.join(folder, "final{}.png".format(limit))
            if not os.path.exists(img_filename):
                print(img_filename)
                cmd = "montage {} -pointsize 8.5 -geometry +4+4 {}".format(filenames, img_filename)
                subprocess.call(cmd, shell=True)

        if not os.path.exists(img_filename):
            continue

        cat = {}
        hash_matrix_filename = os.path.join(folder, "csv", "hashmatrix.npy")
        hash_matrix = np.load(hash_matrix_filename)
        #cat.update(powerlawplot(hash_matrix, folder))
        freqs.append(cat)
        images.append(img_filename)
        print(img_filename)
        c = OrderedDict()
        c.update(model_details)
        K = hash_matrix_to_int(hash_matrix)
        distrib = dict(
            mean=K.mean(),
            var=K.var(ddof=1),
            skew=skew(K)
        )
        c.update(distrib)

        filename = os.path.join(folder, "csv", "nc.csv")
        if not os.path.exists(filename):
            import pandas as pd
            filenames = glob.glob(os.path.join(folder, 'final', '*.png'))
            filenames = sorted(filenames)
            K = {}
            for i, h in enumerate(hm):
               if h not in K:
                   K[h] = i
            indices = K.values()
            filenames = [filenames[ind] for ind in indices]
            print('comute neigh in {}'.format(len(filenames)))
            nc = neighbcorr_filenames(filenames)
            pd.Series([nc]).to_csv(filename)
            print(filename, nc)
        else:
            nc = float(pd.read_csv(filename).columns[1])
        c['neighbcorr'] = nc
        captions.append(c)
    print(len(images))

    indices = range(len(images))
    indices = sorted(indices, key=lambda i: captions[i].items())
    images = [images[i] for i in indices]
    captions = [captions[i] for i in indices]
    freqs = [freqs[i] for i in indices]

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


    if nbpages == -1:
        per_page = 1
    else:
        per_page = len(images) / nbpages
    first = 0
    pg = 1
    mkdir_path(os.path.join(out_folder, where, "generated"))
    plot_names = freqs[0].keys()
    for a in plot_names:
        mkdir_path(os.path.join(out_folder, where, a))
    nb = len(images)

    def save_imgs(first, last, pg=0, w=1500, h=1500, wp=800, hp=800):
        cur_images = images[first:last]
        cur_captions = captions[first:last]
        cur_images = ["\( {} -set label '{}' \)".format(img, caption) for img, caption in zip(cur_images, cur_captions)]
        cur_images = " ".join(cur_images)
        out = os.path.join(out_folder, where, "generated", "page{:04d}".format(pg))
        if w is not None and h is not None:
            sz = '{}x{}'.format(w, h)
        else:
            sz = ''
        # TODO if nb items is 1 just copy the image
        cmd = "montage {} -geometry {}+50+1 {}.png".format(cur_images, sz, out)
        print(cmd)
        subprocess.call(cmd, shell=True)

        cur_freqs = freqs[first:last]

        for p in plot_names:
            cur_images = ["\( {} -set label '{}' \)".format(img[p], caption) for img, caption in zip(cur_freqs, cur_captions)]
            cur_images = " ".join(cur_images)
            out = os.path.join(out_folder, where, p, "page{:04d}".format(pg))
            if wp is not None and hp is not None:
                sz = '-geometry {}x{}'.format(wp, hp)
            else:
                sz = ''
            cmd = "montage {} {}.png".format(cur_images, sz, out)
            print(cmd)
            subprocess.call(cmd, shell=True)

    while first < nb:
        print("page {}".format(pg))
        last = first + per_page
        save_imgs(first, last, pg=pg, w=None, h=None, wp=None, hp=None)
        pg += 1
        first = last
