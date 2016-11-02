from lightjob.cli import load_db
import sys
import os
sys.path.append('/home/mcherti/work/code/feature_generation')
from tools.common import to_generation as to_generation_
from tools.common import preprocess_gen_data
import numpy as np
import click
import joblib
from keras.models import model_from_json
from tools.common import disp_grid
from skimage.io import imsave
from skimage.transform import resize
import shutil

fonts = np.load('/home/mcherti/work/data/fonts/fonts.npz')
fonts_X = fonts['X']
fonts_y = fonts['y']
fonts_examples = [1 - fonts_X[fonts_y==c][0] / 255. for c in range(26)]
fonts_examples = [resize(x[0], (28, 28), preserve_range=True)[None, :, :] for x in fonts_examples]

def get_model(model_folder):
    arch = (open(os.path.join(model_folder, 'model.json'))).read()
    model = model_from_json(arch)
    model.load_weights(os.path.join(model_folder, 'model.pkl'))
    return model

@click.command()
@click.option('--where', default='', required=False)
@click.option('--field', default='stats.out_of_the_box_classification.fonts.objectness', required=False)
@click.option('--model-folder', default='tools/models/external/fonts', required=False)
@click.option('--name', default='', required=False)
@click.option('--nb', default=10, required=False)
@click.option('--images-folder', default='jobs/results/{summary}/images.npz', required=False)
@click.option('--out', default='exported_data/figs/obox/all/{summary}.png', required=False)
@click.option('--to-generation/--no-to-generation', default=True, required=False)
def main(where, field, model_folder, name, nb, images_folder, out, to_generation):
    db = load_db()
    kw = {}
    if where:
        kw['where'] = where
    jobs = db.jobs_with(state='success', **kw)
    if to_generation:
        jobs_gen = to_generation_(jobs)
    else:
        jobs_gen = jobs
    indices = np.arange(len(jobs))
    if field:
        objectness = map(lambda j:db.get_value(j, field, if_not_found=np.nan), jobs_gen)
        objectness = np.array(objectness)
        indices = filter(lambda ind:not np.isnan(objectness[ind]), indices)
        indices = sorted(indices, key=lambda i:objectness[i])
        indices = indices[::-1]
        jobs_gen = [jobs_gen[ind] for ind in indices]
    if nb: jobs_gen = jobs_gen[0:nb]
    model = get_model(model_folder)
    kw = {}
    if name:
        kw['field'] = name
    generate_images(model, jobs_gen, images_folder, out, **kw)

def generate_images(model, jobs_gen, image_folder, out, **kw):
    for ind in range(len(jobs_gen)):
        j = jobs_gen[ind]
        if not j:
            continue
        print(j['summary'])
        filename = image_folder.format(summary=j['summary'])
        if not os.path.exists(filename):
            continue
        X = joblib.load(filename)
        X = preprocess_gen_data(X)
        batch_size = 2048
        preds = []
        for i in range(0, len(X), batch_size):
            x = X[i:i+batch_size]
            preds.append(model.predict(x))
        preds = np.concatenate(preds, axis=0)
        if preds.shape[1] == 36:
            preds = preds[:, 10:]
        nb_classes = preds.shape[1]
        img = construct_image(X, preds, nbrows=6, nbcols=6, border=1, space=10, size=4, nb_classes=nb_classes)
        filename = out.format(summary=j['summary'], i=ind, **kw)
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        print(filename)
        imsave(filename, img)

def construct_image(X, preds, nb_classes=36, nbrows=6, nbcols=6, border=1, space=10, size=4):
    SIZE = size
    s = (28+border) * size + border
    img_all = np.ones((nbrows * (s+space), nbcols * (s+space)  ))
    for i in range(nbrows):
        for j in range(nbcols):
            indices = np.arange(len(X))
            if i*nbcols+j>=nb_classes:
                break
            p=preds[:, i*nbcols+j]        
            indices = np.arange(len(X))
            indices = indices[np.argsort(p[indices])[::-1]]
            img = X[indices]
            imgs = fonts_examples[i*nbcols+j][None, :, :]
            imgs = np.concatenate((imgs, img[0:SIZE*SIZE - 1]), axis=0)
            img = disp_grid(imgs, border=border, bordercolor=(0.3, 0, 0), shape=(SIZE,SIZE))
            img_all[i*(s+space):i*(s+space) + s, j*(s+space):j*(s+space) + s] = img[:,:,0]
    img_all = 1 - img_all
    return img_all

if __name__ == '__main__':
    main()
