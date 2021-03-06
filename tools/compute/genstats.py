import sys
import os
sys.path.append(os.path.dirname(__file__)+"/..")
from common import preprocess_gen_data, compute_objectness, softmax, compute_sample_objectness, compute_objectness_renyi, compute_sample_objectness_renyi, load_model, compute_objectness_correction
from sklearn.cluster import MeanShift
import json
from collections import defaultdict, OrderedDict
from itertools import product
import os
import pandas as pd
import matplotlib as mpl
import glob
import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from scipy.stats import skew
import pickle
import sys
import logging
import intdim_mle
import manifold
import time
import joblib
from toolz.functoolz import memoize
from keras.models import model_from_json

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

mem = joblib.Memory('.cache')

def genstats(jobs, db, force=False, n_jobs=-1, filter_stats=None):
    stats = Parallel(n_jobs=n_jobs)(delayed(compute_stats)(dict(j), force=force, filter_stats=filter_stats)
                                    for j in jobs)
    #stats = [compute_stats(dict(j), force=force, filter_stats=filter_stats) for j in jobs]
    for j, s in zip(jobs, stats):
        update_stats(j, s, db)

def update_stats(job, stats, db):
    db.job_update(job["summary"], dict(stats=stats))

out_of_the_box_names = [
    'mnist_classifier', 
    '5_vs_fake_jobset75',
    'm2', # another mnist classifier (the one wich worked better for jobset75)
    'fonts',
    'letterness',
    'fonts2',
    'fonts3',
]
out_of_the_box_models = [
    'tools/models/external/mnist_classifier', 
    'tools/models/external/5_vs_fake_jobset75',
    'tools/models/mnist/m2',
    'tools/models/external/fonts',
    'tools/models/external/fonts_and_digits',
    'tools/models/external/fonts_multilabel',
    'tools/models/external/fonts2',
]

def compute_stats(job, force=False, filter_stats=None):
    t = time.time()
    j = job
    folder = "jobs/results/{}".format(j['summary'])
    hash_matrix_filename = os.path.join(folder, "csv", "hashmatrix.npy")
    if os.path.exists(hash_matrix_filename):
        hash_matrix = np.load(hash_matrix_filename)
        x = hash_matrix_to_int(hash_matrix)
    else:
        hash_matrix = None
        x = None
        #logger.warning('No hash matrix out there, you are probably using a trainong job type')
    stats = j.get("stats", {})
    if stats is None:
        stats = {}
    s = job['summary']
    if 'ref_job' in j:
        ref_job = j['ref_job']['summary']
    else:
        logger.warning('No ref_job here, you are probably using a training job type')
        ref_job = s

    if filter_stats is not None:
        filter_stats = set(filter_stats)
    def should_compute(s, stats):
        if filter_stats is None:
            if force:
                return True
            if s in stats:
                return False
            return True
        else:
            if s not in filter_stats:
                return False
            if force:
                return True
            if s in stats:
                return False
            return True
    if should_compute('mean', stats):
        logger.info('compute mean of {}'.format(s))
        stats["mean"] = x.mean()
        stats['mean'] = stats['mean'] / len(hash_matrix)
    
    if should_compute('var', stats):
        logger.info('compute var of {}'.format(s))
        stats["var"] = x.var(ddof=1)
        maxvar = ((10000 - 1 + 1)**2 - 1) / 12# variance of max entropy distribution (discrete uniform between 1 and 10000)
        stats['var'] = stats['var'] / maxvar

    if should_compute('skew', stats):
        logger.info('compute skew of {}'.format(s))
        stats["skew"] = skew(x)

    if should_compute('multiplecorrelation', stats):
        logger.info('compute multiplecorrelation of {}'.format(s))
        stats["multiplecorrelation"] = compute_neighcorr(folder, hash_matrix, formula='multiplecorrelation')

    if should_compute('clusdiversity', stats):
        logger.info('computing diversity score using clustering of {}'.format(s))
        stats["clusdiversity"] = compute_clusdiversity(folder, hash_matrix)
        maxdist = np.sqrt(784)# we have binary images, so euclidean dist between full zero vector minus full one vector
        stats["clusdiversity"] = stats["clusdiversity"] / maxdist

    if should_compute('intdim_mle', stats):
        logger.info('computing intdim_le of {}'.format(s))
        stats["intdim_mle"] = compute_intdim(folder, hash_matrix, method='mle')

    if should_compute('convergence_speed', stats):
        logger.info('computing convergence speed of {}'.format(s))
        stats['convergence_speed'] = compute_convergence_speed(folder, j)

    if should_compute('fonts_rec_error', stats):
        logger.info('computing reconstruction error on fonts'.format(s))
        stats['fonts_rec_error'] = compute_rec_error(j, 'fonts', ref_job)

    if should_compute('fontness', stats):
        logger.info('compute fontness of generated data')
        scores = compute_modelness(folder, j, 'discriminators/fonts_64x64.pkl')
        stats['fontness'] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            '10per': float(np.percentile(scores, 10)),
            '90per': float(np.percentile(scores, 90))
        }

    if should_compute('aa_fontness', stats):
        logger.info('compute auto-encoder fontness')
        scores = compute_aa_modelness(folder, j, 'training/fonts/fonts9/model.pkl')
        baseline = 9.29135704041 / (28.*28.)
        scores /= baseline
        stats['aa_fontness'] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            '10per': float(np.percentile(scores, 10)),
            '90per': float(np.percentile(scores, 90))
        }
    if should_compute('tsne', stats):
        logger.info('compute tsne...')
        stats['tsne'] = compute_tsne(folder, hash_matrix)
    if should_compute('knn_classification_accuracy', stats):
        logger.info('compute knn classification accuracy')
        stats['knn_classification_accuracy'] = compute_knn_classification_accuracy(folder, hash_matrix, ref_job)
    if should_compute('training', stats):
        logger.info('compute training stats')
        stats['training'] = compute_training_stats(folder, ref_job)


    if should_compute('out_of_the_box_classification', stats):
        stat = {}
        for model_name, model_folder in zip(out_of_the_box_names, out_of_the_box_models):
            a = time.time()
            stat[model_name] = compute_out_of_the_box_classification(folder, model_name, model_folder)
        stats['out_of_the_box_classification'] = stat

    if should_compute('feature_space_eval', stats):
        logger.info('compute feature space eval')
        stats['feature_space_eval'] = compute_feature_space_eval(folder)

    if should_compute('parzen_digits', stats):
        logger.info('compute parzen digits')
        stats['parzen_digits'] = compute_parzen_digits(folder)

    if should_compute('parzen_letters', stats):
        logger.info('compute parzen letters')
        stats['parzen_letters'] = compute_parzen_letters(folder)

    delta_t = time.time() - t
    logger.info('Finished on {}, took {:.3f}'.format(j['summary'], delta_t))
    return stats

def construct_data(job_folder, hash_matrix, transform=lambda x:x):
    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    filenames = sorted(filenames)
    indices = unique_indices(hash_matrix)
    filenames = [filenames[ind] for ind in indices]
    if len(filenames) == 0:
        return None
    X = []
    for im in get_images(filenames):
        X.append([transform(im)])
    X = np.concatenate(X, axis=0)
    X = X.reshape((X.shape[0], -1))
    return X

@memoize
def load_mnist():
    from datakit import mnist
    data = mnist.load(which='test')
    mnist_test_X = data['test']['X'] / 255.
    mnist_test_X = mnist_test_X.astype(np.float32)
    return mnist_test_X

@memoize
def load_fonts():
    from tools.common import resize_set
    import os
    fonts = np.load('/home/mcherti/work/data/fonts/fonts.npz')
    fonts_X = fonts['X']
    fonts_y = fonts['y']
    X = 1 - fonts_X / 255.
    X = resize_set(X, 28, 28)
    print(X.shape)
    return X

def compute_parzen_digits(folder, **kw):
    return _parzen(folder, load_mnist(), **kw)

def compute_parzen_letters(folder, **kw):
    return _parzen(folder, load_fonts(), **kw)

def _parzen(folder, X, nb_samples=None, batch_size=10):
    from likelihood_estimation_parzen import ll_parzen
    data = joblib.load(os.path.join(folder, 'images.npz'))
    data = preprocess_gen_data(data)
    if nb_samples:
        data = data[0:nb_samples]
    sigma = 0.23
    ll = ll_parzen(sigma=sigma, samples=flatten(data), test_X=flatten(X), batch_size=batch_size)
    ll = np.array(ll)
    stat = {'mean': float(ll.mean()), 'std': float(ll.std() / (batch_size**2)) }
    return stat

def flatten(x):
    return x.reshape((x.shape[0], -1))

def compute_out_of_the_box_classification(folder, model_name, model_folder, nb_samples=None):
    preds_filename = "{}/out_of_the_box_classification_{}.npz".format(folder, model_name)
    if not os.path.exists(preds_filename) or True:
        data = joblib.load(os.path.join(folder, 'images.npz'))
        data = preprocess_gen_data(data)
        model = load_discr_model(model_folder)
        try:
            if data.max() >= 1:
                data = data / float(data.max())
            if nb_samples:
                data = data[0:nb_samples]
            pred = model.predict(data)
        except Exception as ex:
            print('Error on : {}, {}.Skip.'.format(folder, ex))
            return {}
        joblib.dump(pred, preds_filename, compress=9)
    else:
        pred = joblib.load(preds_filename)
    stats = {}
    stats['objectness'] = compute_objectness(pred)
    stats['objectness_correction'] = compute_objectness_correction(pred)
    stats['sample_objectness'] = float(compute_sample_objectness(pred).mean())
    stats['objectness_renyi'] = compute_objectness_renyi(pred)
    stats['sample_objectness_renyi'] = float(compute_sample_objectness_renyi(pred).mean())
    pred_sorted = np.sort(pred, axis=1)[:, ::-1]
    for k in range(pred_sorted.shape[1]):
        stats['top{k}_prediction'.format(k=k + 1)] = float(pred_sorted[:, k].mean())
    if model_name == 'letterness':
        t = time.time()
        data = joblib.load(os.path.join(folder, 'images.npz'))
        data = preprocess_gen_data(data)
        if nb_samples:data = data[0:nb_samples]
        model = load_discr_model(model_folder, linear=True)
        pred = model.predict(data)
        digits = np.arange(0, 10)
        letters = np.arange(10, 10 + 26)
        stats['objectness_digits'] = compute_objectness(softmax(pred[:, digits]))
        stats['objectness_letters'] = compute_objectness(softmax(pred[:, letters]))
        stats['sample_objectness_digits'] = float(compute_sample_objectness(softmax(pred[:, digits])).mean())
        stats['sample_objectness_letters'] = float(compute_sample_objectness(softmax(pred[:, letters])).mean())
        stats['max_letters'] = float(softmax(pred)[:, letters].max(axis=1).mean())
        ent = entropy(probas_from_occurences(softmax(pred)[:, letters].argmax(axis=1))) / np.log(len(letters))
        stats['diversity_letters'] = ent
        stats['diversity_max_letters'] = 0.5 * stats['max_letters'] + 0.5 * ent
        stats['diversity_max_letters_alpha8'] = 0.8 * stats['max_letters'] + 0.2 * ent
        stats['diversity_max_letters_alpha9'] = 0.9 * stats['max_letters'] + 0.1 * ent
        stats['max_digits'] = float(softmax(pred)[:, digits].max(axis=1).mean())
        ent = entropy(probas_from_occurences(softmax(pred)[:, digits].argmax(axis=1))) / np.log(len(digits))
        stats['diversity_digits'] = ent
        stats['diversity_max_digits'] = stats['max_digits'] * 0.5  + ent * 0.5
        stats['diversity_max_digits_alpha8'] = stats['max_digits'] * 0.8  + ent * 0.2
        stats['diversity_max_digits_alpha9'] = stats['max_digits'] * 0.9  + ent * 0.1
        pred_proba = softmax(pred)
        pred_proba_letters = pred_proba[:, letters]
        pred_proba_digits = pred_proba[:, digits]
        for theta in (0.85, 0.9, 0.95, 0.98, 0.99, 0.999):
            s = str(theta).replace('0.', '')
            # letters
            selected_letters = (pred_proba_letters.max(axis=1) > theta)
            stats['count_letters_{}'.format(s)] = float(selected_letters.mean())
            # letters with diversity term
            ent = entropy(probas_from_occurences(pred_proba_letters[selected_letters].argmax(axis=1))) / np.log(pred_proba_letters.shape[1])
            stats['diversity_count_letters_{}'.format(s)] = 0.5 * selected_letters.mean() + 0.5 * ent
            stats['diversity_count_letters_alpha8_{}'.format(s)] = 0.8 * selected_letters.mean() + 0.2 * ent
            stats['diversity_count_letters_alpha9_{}'.format(s)] = 0.9 * selected_letters.mean() + 0.1 * ent
            print('count_letters', selected_letters.mean(), ent)
            # digits
            selected_digits = (pred_proba_digits.max(axis=1) > theta)
            stats['count_digits_{}'.format(s)] = float(selected_digits.mean())
            # digits with diversity
            ent = entropy(probas_from_occurences(pred_proba_digits[selected_digits].argmax(axis=1))) / np.log(pred_proba_digits.shape[1])
            stats['diversity_count_digits_{}'.format(s)] = 0.5 * selected_digits.mean() + 0.5 * ent
            stats['diversity_count_digits_alpha8_{}'.format(s)] = 0.8 * selected_digits.mean() + 0.2 * ent
            stats['diversity_count_digits_alpha9_{}'.format(s)] = 0.9 * selected_digits.mean() + 0.1 * ent

    return stats

def entropy(x):
    return -np.dot(x, np.log(x))

def probas_from_occurences(x):
    cnt = Counter(x)
    total = float(len(x))
    probas = []
    for k, nb in cnt.items():
        p = nb/total
        probas.append(p)
    return probas

@memoize
def load_discr_model(model_folder, linear=False):
    js = open(os.path.join(model_folder, 'model.json')).read()
    if linear:
        js = js.replace('softmax', 'linear')
    model = model_from_json(js)
    try:
        model.load_weights(os.path.join(model_folder, 'model.pkl'))
    except Exception:
        model.load_weights(os.path.join(model_folder, 'model.h5'))
    return model

def compute_training_stats(folder, ref_job):
    from tasks import load_
    filename = "jobs/results/{}/model.pkl".format(ref_job)
    logger.info('loading {}'.format(filename))
    if not os.path.exists(filename):
        return {}
    _, info = load_(filename)
    logger.info('loaded {}'.format(filename))
    stats = info['stats']
    stats_dict = {}
    keys = stats[0].keys()
    for k in keys:
        stats_dict[k] = [s[k] for s in stats if k in s]
        n = len(stats_dict[k])
        d = {}
        d['5per'] = float(stats_dict[k][int(n*0.05)])
        d['10per'] = float(stats_dict[k][int(n*0.1)])
        d['30per'] = float(stats_dict[k][int(n*0.3)])
        d['50per'] = float(stats_dict[k][int(n*0.5)])
        d['70per'] = float(stats_dict[k][int(n*0.7)])
        d['90per'] = float(stats_dict[k][int(n*0.9)])
        stats_dict[k + '_details'] = d
        stats_dict[k] = float(stats_dict[k][-1])
    return stats_dict

def compute_modelness(job_folder, hash_matrix, model_path):
    import pickle
    from lasagne import layers
    import theano.tensor as T
    from skimage.transform import resize
    import theano
    print('load data...')
    data = pickle.load(open(model_path))
    discr = data['discriminator']
    discr_weights = data['discriminator_weights']
    layers.set_all_param_values(discr, discr_weights)

    print('compile function...')
    X = T.tensor4()

    get_score = theano.function([X], layers.get_output(discr,X))

    input_layer = layers.get_all_layers(discr)[0]
    def transform(x):
        c, w, h = input_layer.output_shape[1:]
        if x.shape[1] != w or x.shape[2] != h:
            return resize(x, (w, h), preserve_range=True)
        else:
            return x
    print('create data...')
    imgs = construct_data(job_folder, hash_matrix, transform=transform)
    imgs -= imgs.min()
    imgs /= imgs.max()
    print(imgs.max(), imgs.min())
    shape = (imgs.shape[0],) + input_layer.output_shape[1:]
    imgs = imgs.reshape(shape)
    imgs = imgs.astype(np.float32)
    print('compute scores...')
    scores = minibatcher(imgs, get_score, size=1000)[:, 0]
    return scores

def compute_aa_modelness(job_folder, hash_matrix, model_path):
    import pickle
    from lasagne import layers
    import theano.tensor as T
    from skimage.transform import resize
    import theano

    from tasks import check
    from data import load_data
    from lasagnekit.easy import iterate_minibatches

    v = check(what="notebook",
              filename=model_path,
              dataset=None,
              force_w=28,
              force_h=28,
              force_c=1)
    capsule, data, layers, w, h, c = v

    imgs = construct_data(job_folder, hash_matrix)
    imgs -= imgs.min()
    imgs /= imgs.max()
    shape = (imgs.shape[0], c, w, h)
    print(shape)
    imgs = imgs.reshape(shape)
    imgs = imgs.astype(np.float32)
    print('compute scores...')
    scores = minibatcher(imgs, lambda x:((capsule.reconstruct(x)-x)**2).mean(axis=(1, 2, 3)), size=1000)
    return scores

def minibatcher(X, f, size=128):
    from lasagnekit.easy import iterate_minibatches
    res = []
    for sl in iterate_minibatches(X.shape[0], size):
        r = f(X[sl])
        res.append(r)
    return np.concatenate(res, axis=0)

def compute_tsne(job_folder , hash_matrix):
    from tasks import check
    from data import load_data
    from lasagnekit.easy import iterate_minibatches
    from sklearn.manifold import TSNE
    from lasagne.layers.helper import get_output
    import pandas as pd
    import theano.tensor as T
    import theano
    import cPickle as pickle
    filenames = []
    try:
        nb = 1000
        v = check(what="notebook",
                  filename="training/initial_models/model_E.pkl",
                  dataset='digits')
        capsule, data, layers, _, _, _ = v

        np.random.seed(2)
        data_X = data.train.X
        data_y = data.train.y
        data_indices = np.arange(0, len(data_X))
        data_indices = data_indices[0:nb]
        data_X = data_X[data_indices]
        data_y = data_y[data_indices]
        data_X -= data_X.min()
        data_X /= data_X.max()

        np.random.seed(2)
        X = construct_data(job_folder, hash_matrix)
        indices = np.arange(0, len(X))
        np.random.shuffle(indices)
        indices = indices[0:nb]
        X = X[indices]
        X -= X.min()
        X /= X.max()
        X_full = np.concatenate((X, data_X), axis=0)
        cats = np.array([-1] * len(X) + data_y.tolist())
        # input space
        np.random.seed(2)
        tsne = TSNE(perplexity=15, early_exaggeration=20, verbose=1, n_components=2)
        X_2d = tsne.fit_transform(X_full)
        filename = '{}/tsne_input.pkl'.format(job_folder)
        input_filename = filename
        filenames.append(filename)
        dt = {'x':X_2d[:, 0],
              'y': X_2d[:, 1],
              'gen_ind': indices,
              'dataset_ind': data_indices,
              'images': X_full,
              'categories': cats}
        fd = open(filename, 'w')
        pickle.dump(dt, fd)
        fd.close()

        # latent space
        x = T.tensor4()
        fn = theano.function([x], get_output(layers['conv3'], x))
        X = X.astype(np.float32)
        X_full = X_full.astype(np.float32)
        feats = fn(X_full.reshape((X_full.shape[0], 1, 28, 28)))
        feats = feats.max(axis=(2, 3))
        feats = feats.reshape((feats.shape[0], -1))
        np.random.seed(2)
        tsne = TSNE(perplexity=15, early_exaggeration=20, verbose=1, n_components=2)
        X_2d = tsne.fit_transform(feats)
        filename = '{}/tsne_latent.pkl'.format(job_folder)
        latent_filename = filename
        filenames.append(filename)
        dt = {'x':X_2d[:, 0],
              'y': X_2d[:, 1],
              'gen_ind': indices,
              'dataset_ind': data_indices,
              'images': X_full,
              'categories': cats}

        fd = open(filename, 'w')
        pickle.dump(dt, fd)
        fd.close()

        print('OK DONE')
        return {'input': input_filename, 'latent': latent_filename}
    except Exception as ex:
        print(str(ex))
        return {}
def compute_rec_error(job, dataset, ref_job):
    from tasks import check
    from data import load_data
    from lasagnekit.easy import iterate_minibatches

    v = check(what="notebook",
              filename="jobs/results/{}/model.pkl".format(ref_job),
              dataset='digits') # any would work, we dont care
    capsule, data, layers, w, h, c = v
    data = load_data(dataset, w=w, h=h)
    assert hasattr(data, 'train')
    X = data.train.X
    rec_errors = []
    for batch in iterate_minibatches(X.shape[0], batchsize=1000):
        rec_error = ((capsule.preprocess(X[batch]) - capsule.reconstruct(capsule.preprocess(X[batch])))**2).mean(axis=(1, 2, 3))
        rec_errors.append(rec_error)
    baseline = 9.29135704041 / (28.*28.)
    return float(np.concatenate(rec_errors, axis=0).mean()) / baseline

def compute_convergence_speed(job_folder, job):
    max_nb_iterations = 100.
    speed = 1. - (len(open(os.path.join(job_folder, "csv", "iterations.csv")).readlines()) - 1) / max_nb_iterations
    # speed = 0 (worst) 1(best)
    return speed

def compute_almost_uniq(folder, hash_matrix):
    bandwidth = None # meaning it is estimated from data
    X = construct_data(folder, hash_matrix)
    try:
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(X)
        return len(ms.cluster_centers_) / float(len(X))
    except Exception:
        return np.nan

def compute_knn_classification_accuracy(job_folder, hash_matrix, ref_job):
    from sklearn.neighbors import KNeighborsClassifier as KNN
    from lasagnekit.easy import iterate_minibatches
    from tasks import check
    import time
    t = time.time()

    max_iter = 100
    n_neighbors =(5,)
    v = check(what="notebook",
              filename="jobs/results/{}/model.pkl".format(ref_job),
              kw_load_data={'include_test': True},
              dataset="digits")
    capsule, data, layers, w, h, c = v
    X_train, y_train = data.train.X, data.train.y
    knns = {}
    for nn in n_neighbors:
        knn = KNN(n_neighbors=nn, n_jobs=-1).fit(X_train, y_train)
        knns[nn] = knn
    X_test = data.test.X
    y_test = data.test.y
    accs = defaultdict(list)

    for nn in n_neighbors:
        logger.info('nn:{}'.format(nn))
        for batch in iterate_minibatches(X_test.shape[0], batchsize=512):
            logger.info('X_test : {}'.format(batch))
            x = X_test[batch]
            x = capsule.preprocess(x)
            logger.info('iterative refinement...')
            for i in range(max_iter):
                x_prev = x
                x = capsule.reconstruct(x)
                x = x > 0.5
                x = x.astype(np.float32)
                score = np.abs(x - x_prev).sum()
                print(score)
                if score == 0:
                    break
            logger.info('performed {} iterations'.format(i + 1))
            x = x.reshape((x.shape[0], -1))
            logger.info('knn prediction...')
            y_pred = knn.predict(x)
            accs[nn].extend((y_test[batch] == y_pred).tolist())
        logger.info('Acccuracy on {} : {}'.format(nn, np.mean(accs[nn])))
    for k, v in accs.items():
        accs[k] = float(np.mean(v))
    accs = dict(accs)
    print(accs)
    logger.info(time.time() - t)
    return accs

def compute_feature_space_eval(job_folder):
    from datakit.mnist import load
    from sklearn.linear_model import LogisticRegression
    import theano
    import theano.tensor as T
    import lasagne
    data = load(which='all')
    X_train = data['train']['X'] / 255.
    X_train = X_train.astype(np.float32)
    y_train = data['train']['y']
    X_test = data['test']['X'] / 255.
    X_test = X_train.astype(np.float32)
    y_test = data['test']['y']
    model, _, layers = load_model(job_folder + '/model.pkl')
    X = T.tensor4()
    lays = [l for l in layers if l.startswith('conv') or l.startswith('hid')]
    if len(lays) == 0:
        return {}
    lays = sorted(lays)
    print(lays)
    hid_layer = lays[-1]
    hid = lasagne.layers.get_output(layers[hid_layer], X)
    hid = hid.reshape((hid.shape[0], -1))
    get_hidden = theano.function([X], hid)
    h_train = minibatcher(X_train, get_hidden, size=1024)
    
    clf = LogisticRegression()
    mean_h = h_train.mean(axis=0, keepdims=True)
    std_h = h_train.std(axis=0, keepdims=True)
    eps = 1e-8
    h_train = (h_train - mean_h) / (std_h + eps)
    clf.fit(h_train, y_train)
    h_test = minibatcher(X_test, get_hidden, size=1024)
    h_test = (h_test - mean_h) / std_h
    acc = (clf.predict(h_test) == y_test).mean()
    acc = float(acc)
    print('test accuracy : {}'.format(acc))
    return {'test_accuracy': acc}

def compute_manifold_dist(job_folder, hash_matrix, ref_job):

    import numpy as np
    from tasks import check
    import theano
    import theano.tensor as T
    import lasagne

    v = check(what="notebook",
              filename="jobs/results/{}/model.pkl".format(ref_job),
              dataset="digits")
    capsule, data, layers, w, h, c = v

    np.random.seed(1234)
    nb = 10000
    Y = data.train.X[np.random.choice(np.arange(len(data.train.X)), size=nb, replace=False)]
    x = T.tensor4()
    g = theano.function(
        [x],
        lasagne.layers.get_output(layers['hid'], x)
    )
    H = g(Y.reshape((Y.shape[0], c, w,h)))

    NN = 100 # nb of  neighbors
    k = 10 # logk defines the entropy of tau_Y and tau_Z distributions
    ks = 5
    dist, _ = manifold.compute_dist(Y, H, NN=NN, k=k, ks=ks,
                                    compute_density=False,
                                    nb_integrations=1, # repeat if you want to estimate variance of the integral estimation
                                    nb_iter=100,
                                    integration_method='monte_carlo')
    return np.mean(dist)

def compute_dpgmmclusdiversity(job_folder, hash_matrix):
    from sklearn.mixture import DPGMM
    X = construct_data(job_folder, hash_matrix)
    clus = DPGMM(n_components=1000, verbose=1)
    clus.fit(X)
    n_components = clus.n_components
    print('Nb of components : ', n_components)
    return n_components


def compute_clusdiversity(job_folder, hash_matrix, nb_clusters=1000):
    from sklearn.cluster import KMeans
    X = construct_data(job_folder, hash_matrix)
    if X.shape[0] < nb_clusters:
        nb_clusters = max(X.shape[0] / 10, 1)
    logger.info(str(nb_clusters))
    clus = KMeans(n_clusters=nb_clusters, verbose=1, n_jobs=1, n_init=2)
    clus.fit(X)
    # dists from closest cluster center
    dists = clus.transform(X).min(axis=1)
    # avg distance of points to their closest cluster center
    return dists.mean()

def compute_intdim(job_folder, hash_matrix, method='mle'):
    import numpy as np
    np.random.seed(42)
    X = construct_data(job_folder, hash_matrix)
    nb = min(100, X.shape[0])
    X = X[np.random.choice(np.arange(len(X)), size=nb, replace=False)]
    k1 = 10 # start of interval(included)
    k2 = 20 # end of interval(included)
    intdim_k_repeated = intdim_mle.repeated(
        intdim_mle.intrinsic_dim_scale_interval,
        X,
        mode='bootstrap',
        nb_iter=100, # nb_iter for bootstrapping
        verbose=1,
        k1=k1, k2=k2)
    print('Estimate : {} +/- {}'.format(np.mean(np.mean(intdim_k_repeated, axis=1)), np.var(np.mean(intdim_k_repeated, axis=1))))
    return np.mean(intdim_k_repeated)

def compute_nearestneighbours_diversity(job_folder, hash_matrix, nearest_neighbors=100):
    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    X = []
    for im in get_images(filenames):
        X.append([im])
    X = np.concatenate(X, axis=0)
    X = X.reshape((X.shape[0], -1))
    dist = nearestneighbours_distance(X, nearest_neighbors=nearest_neighbors)
    return dist.mean()



def nearestneighbours_distance(X, nearest_neighbors=8):
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=nearest_neighbors, algorithm='ball_tree').fit(X)
        distances, _ = nbrs.kneighbors(X)
        return distances.mean(axis=1)


def compute_neighcorr(job_folder, hash_matrix, formula='mine'):
    filenames = glob.glob(os.path.join(job_folder, 'final', '*.png'))
    filenames = sorted(filenames)
    indices = unique_indices(hash_matrix)
    filenames = [filenames[ind] for ind in indices]
    if formula == 'mine':
        nc = neighbcorr_filenames_mine(filenames)
    elif formula == 'multiplecorrelation':
        nc = neighbcorr_filenames_multiplecorrelation(filenames)
    return nc

def unique_indices(hm):
    K = {}
    for i, h in enumerate(hm):
        if h not in K:
            K[h] = i
    return K.values()

def neighbcorr_filenames_multiplecorrelation(filenames, pad=3):
    # https://en.wikipedia.org/wiki/Multiple_correlation
    from sklearn.linear_model import LinearRegression
    xdata = defaultdict(list)
    ydata = defaultdict(list)
    for im in get_images(filenames):
        for x in range(pad, im.shape[0] - pad):
            for y in range(pad, im.shape[1] - pad):
                pxc = im[x, y]
                dt = []
                for dx, dy in product((0, 1, -1), (0, 1, -1)):
                    if dx == 0 and dy == 0:
                        continue
                    px = im[x + dx, y + dy]
                    dt.append(px)
                xdata[(x, y)].append(dt)
                ydata[(x, y)].append(pxc)
    rsqr_l = []
    for pos in xdata.keys():
        x = xdata[pos]
        y = ydata[pos]
        rsqr = np.sqrt(LinearRegression().fit(x, y).score(x, y))
        rsqr_l.append(rsqr)
    return np.mean(rsqr_l)


def neighbcorr_filenames_mine(filenames):
    corrdata = defaultdict(int)
    for im in get_images(filenames):
        neighcorr(im, corrdata=corrdata, pad=3)
    nc = np.abs(np.array(corrdata.values())).mean() / len(filenames)
    return nc


def get_images(filenames):
    from skimage.io import imread
    for f in filenames:
        im = imread(f)
        im = 2 * (im / im.max()) - 1
        assert set(im.flatten().tolist()) <= set([1, -1]), set(im.flatten().tolist())
        yield im


def neighcorr(im, corrdata=None, pad=3):
    assert corrdata is not None
    i = 0
    for x in range(pad, im.shape[0] - pad):
        for y in range(pad, im.shape[1] - pad):
            pxc = im[x, y]
            for dx, dy in product((0, 1, -1), (0, 1, -1)):
                px = im[x + dx, y + dy]
                c = px * pxc
                corrdata[i] += c
            i += 1

def hash_matrix_to_int(hm):
    cnt = Counter(hm)
    s = np.argsort(cnt.values())[::-1]
    K = cnt.keys()
    K = [K[s[i]] for i in range(len(K))]
    K_to_int = {k: i + 1 for i, k in enumerate(K)}
    x = [K_to_int[v] for v in hm]
    x = np.array(x)
    return x
