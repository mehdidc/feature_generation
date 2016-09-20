from docopt import docopt
from tqdm import tqdm

import numpy as np
from common import load_model, build_brush_func, disp_grid, prop_uniques, seq_to_video

import pandas as pd
from skimage.io import imsave

def main():
    doc = """
    Usage:
        export_img.py [--from-json=JSON] [--id=ID] FILE [OUTPUT]

    Arguments:
        FILE input file
        OUTPUT output directory

    Options:
        -h --help     Show this screen
        --from-json=JSON json configuration file
        --id=ID id of job
    """
    np.random.seed(2)

    args = docopt(doc)
    model, data, layers = load_model(args['FILE'])
    c, h, w = layers['output'].output_shape[1:]

    O = args['OUTPUT'] + '/'
    brush = build_brush_func(layers) # transforms an image to sequence of images
    X = data.X[0:11*11]
    imgs = brush(model.preprocess(X)) # (examples, time, w, h)
    seq_to_video(imgs, O+'seq.mp4')

    im1 = disp_grid(model.preprocess(data.X[0:100]), border=1, bordercolor=(.3, .3, .3))
    im2 = disp_grid(model.reconstruct(model.preprocess(data.X[0:100])), border=1, bordercolor=(.5, 0, 0))
    im_mix = np.empty((im1.shape[0], im1.shape[1] + im2.shape[1], 3))
    im_mix[:, 0:im1.shape[1]] = im1
    im_mix[:, im1.shape[1]:] = im2
    imsave(O + 'im_mix.png', im_mix)
    
    nb_iter = 200
    nb_examples = 100
    thresh = 'moving'
    use_noise = False
    # PREP
    if use_noise: noise = np.random.normal(0, 0.5, size=imgs[:, 0].shape).astype(np.float32) #(for colored images)
    else: noise = 0

    if thresh == 'moving':
        whitepx_ratio = (data.X>0.5).sum() / np.ones_like(data.X).sum()

    imgs = np.empty((nb_examples, nb_iter + 1, c, w, h)) # 1 = color channel
    imgs = imgs.astype(np.float32)
    imgs[:, 0] = np.random.uniform(size=(nb_examples, c, w, h))
    scores = []
    diversities = []
    # ITERATION
    for i in tqdm(range(1, nb_iter + 1)):
        if use_noise:noise = np.random.normal(0, 1, size=imgs[:, 0].shape).astype(np.float32) #(for colored images)
        else:noise = 0
            
        imgs[:, i] = brush(imgs[:, i - 1] + noise)[:,-1]
        if c == 1:
            if thresh == 'moving':
                vals = imgs[:, i].flatten()
                vals = vals[np.argsort(vals)]
                thresh_ = vals[-int(whitepx_ratio * len(vals)) - 1]
            else:
                thresh_ = thresh
            if thresh_:
                imgs[:, i] = imgs[:, i] > thresh_ # binarize
        score = np.abs(imgs[:, i - 1] - imgs[:, i]).sum()
        scores.append(score)
        diversity = prop_uniques(imgs[:, i])
        diversities.append(diversity)
    
    img = disp_grid(imgs[:, -1], border=1, bordercolor=(0.3, 0, 0))
    imsave(O + 'ir.png', img)
    seq_to_video(imgs, O+'ir.mp4', border=0, bordercolor=(0, 0, 0))
    pd.Series(score).to_csv(scores, O+'scores.csv')
    pd.Series(diversity).to_csv(diversity, O+'diversity.csv')
    # OMNIGLOT TEST
    dt_test = load_data('omniglot', w=w, h=h)
    nb = 100
    dt = dt_test.X[0:nb]
    try:
        dt = dt.reshape((nb, c, w, h))
    except Exception:
        dt = dt.reshape((nb, 1, w, h))
        dt = dt * np.ones((1, 3, 1, 1))
        dt = dt.astype(np.float32)
    print(dt.shape)
    rec = model.reconstruct(dt)
    print(((rec - dt)**2).mean())
    im1 = disp_grid(model.preprocess(dt[0:nb]), border=1, bordercolor=(.3, .3, .3))
    im2 = disp_grid(model.reconstruct(model.preprocess(rec[0:nb])), border=1, bordercolor=(.5, 0, 0))
    im_mix = np.empty((im1.shape[0], im1.shape[1] + im2.shape[1], 3))
    im_mix[:, 0:im1.shape[1]] = im1
    im_mix[:, im1.shape[1]:] = im2
    imsave(O+'im_mix_new_dataset.png', im_mix)

if __name__ == '__main__':
    main()
