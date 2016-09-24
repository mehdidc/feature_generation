from docopt import docopt
from tqdm import tqdm

import numpy as np
from common import (
    load_model, build_brush_func, 
    disp_grid, prop_uniques, seq_to_video, 
    build_encoders, build_pointer_images, 
    build_image_to_code_func, build_code_to_image, 
    normalize,
    sigmoid, get_scale, get_bias)

import pandas as pd
from skimage.io import imsave

from helpers import mkdir_path
import theano.tensor as T
import theano

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
    model, data, layers = load_model(args['FILE'], kw_load_data={"nb_examples": 121, "image_collection_mode": "random"})
    c, h, w = layers['output'].output_shape[1:]

    O = args['OUTPUT'] + '/'
    mkdir_path(O)
    
    nb_parallel = model.hypers['model_params'].get('parallel')

    # COMPILE
    encoders = build_encoders(layers, nb_parallel=nb_parallel)
    lays = (['brush_{}'.format(i) for i in range(nb_parallel)] if nb_parallel else ['brush'])
    lays = tuple(lays)
    brush = build_brush_func(
        layers, 
        lay=lays,
        nonlin=lambda x:x)

    X = model.preprocess(data.X[0:11*11])
    # POINTER
    cols = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    codes = []
    for encoder in encoders:
        code = encoder(X)
        codes.append(code)
    pointers = []
    for i, code in enumerate(codes):
        pointer = build_pointer_images(code, cols[i], w, h, p=1)
        pointers.append(pointer)
    pointers = sum(pointers)
    seq_to_video(pointers, O + 'pointers.mp4')

    # BRUSH
    imgs = brush(X) # (examples, time, w, h)
    imgs = sum(imgs)
    bias = get_bias(layers)
    scale = get_scale(layers)
    
    x_t = T.tensor5()
    apply_nonlin = theano.function([x_t], layers['output'].nonlinearity(x_t))
    imgs = (imgs * scale + bias) 
    imgs = apply_nonlin(imgs)
    #imgs = normalize(imgs, axis=(1, 3,4))
    #imgs = imgs + pointers * (1-imgs)
    seq_to_video(imgs, O+'seq.mp4')

    # RECONSTRUCT
    im1 = disp_grid(X, border=1, bordercolor=(.3, .3, .3))
    im2 = disp_grid(model.reconstruct(X), border=1, bordercolor=(.5, 0, 0))
    im_mix = np.empty((max(im1.shape[0], im2.shape[0]), im1.shape[1] + im2.shape[1], 3))
    im_mix[0:im1.shape[0], 0:im1.shape[1]]

    im_mix[0:im1.shape[0], 0:im1.shape[1]] = im1
    im_mix[0:im2.shape[0], im1.shape[1]:] = im2
    imsave(O+'im_mix.png', im_mix)


    # ITERATIVE REFINIMENT

    nb_iter = 200
    nb_examples = 100
    thresh = 'moving'
    use_noise = False
    if c == 3:
        use_noise = True

    # PREP
    if c == 1 and thresh == 'moving':
        whitepx_ratio = (data.X>0.5).sum() / np.ones_like(data.X).sum()

    imgs = np.empty((nb_examples, nb_iter + 1, c, w, h)) # 1 = color channel
    imgs = imgs.astype(np.float32)
    imgs[:, 0] = np.random.uniform(size=(nb_examples, c, w, h))


    if use_noise: noise = np.random.normal(0, 0.5, size=imgs[:, 0].shape).astype(np.float32) #(for colored images)
    else: noise = 0

    scores = []
    diversities = []
    # ITERATION
    for i in tqdm(range(1, nb_iter + 1)):
        #if use_noise:noise = np.random.normal(0, 1, size=imgs[:, 0].shape).astype(np.float32) #(for colored images)
        #else:noise = 0
        imgs[:, i] = model.reconstruct(imgs[:, i - 1] + noise)
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
    pd.Series(score).to_csv(O+'scores.csv')
    pd.Series(diversity).to_csv(O+'diversity.csv')


    # INTERPOLATION
    if nb_parallel:
        lay = ['coord_{}'.format(i) for i in range(nb_parallel)]
    else:
        lay = ['coord']
    lay = tuple(lay)
    tensors = tuple(T.tensor3() for _ in range(nb_parallel))
    img2code = build_image_to_code_func(layers, lay=lay)
    code2img = build_code_to_image(layers, lay=lay, X=tensors)
    
    indices = np.random.randint(0, len(data.X), size=4)
    codes = img2code(model.preprocess(data.X[indices]))
    shapes = [c.shape[1:] for c in codes]
    codes_flat = [c.reshape(c.shape[0], -1) for c in codes]
    sizes = [c.shape[1] for c in codes_flat]
    codes_concat = np.concatenate(codes_flat, axis=1)

    z_dim = codes_concat.shape[1:]
    D = 12
    alpha = np.linspace(0, 1, D)
    beta = np.linspace(0, 1, D)
    grid_codes = np.empty((D*D,) + z_dim, dtype='float32')
    k = 0
    for a in alpha:
        for b in beta:
            grid_codes[k] = a*b*codes_concat[0] + a*(1-b)*codes_concat[1] + (1-a)*b*codes_concat[2]  + (1-a)*(1-b)*codes_concat[3]
            k +=1

    i = 0
    orig_codes = []
    for s in sizes:
        orig_codes.append(grid_codes[:, i:i+s])
        i+=s
    orig_codes = [orig_code.reshape((orig_code.shape[0],) + shape) for orig_code, shape in zip(orig_codes, shapes)]
    print(orig_codes[0].shape)
    grid_imgs = code2img(*orig_codes)
    imsave(O+'grid.png', disp_grid(grid_imgs, border=2, bordercolor=(0.3,0.,0.)))


if __name__ == '__main__':
    main()
