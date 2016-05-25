import matplotlib as mpl
import os
import pandas as pd
if os.getenv("DISPLAY") is None:  # NOQA
    mpl.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import dill
import sys
from invoke import task
from collections import OrderedDict
import theano.tensor as T
import model
import theano
from datetime import datetime
from lasagne import updates
from lasagnekit.easy import (
    make_batch_optimizer, InputOutputMapping,
    build_batch_iterator)
from lasagne import layers as L
from lasagnekit.nnet.capsule import Capsule, make_function
from lasagnekit.misc.plot_weights import dispims_color, tile_raster_images
from lasagnekit.easy import get_stat, iterate_minibatches
import numpy as np
from helpers import salt_and_pepper, zero_masking, bernoulli_sample, minibatcher
from data import load_data
from model import *  # for dill
from skimage.io import imsave
import logging


sys.setrecursionlimit(10000)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from lightjob.cli import get_dotfolder
from lightjob.db import DB, SUCCESS, RUNNING, AVAILABLE, ERROR, PENDING

db_folder = get_dotfolder()

def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


@task
def train(dataset="digits", prefix="",
          model_name="model8",
          budget_hours=np.inf,
          update_db=None,
          force_w=None, force_h=None,
          params=None):
    import json
    mkdir_path("{}/features".format(prefix))
    mkdir_path("{}/recons".format(prefix))
    mkdir_path("{}/out".format(prefix))

    if type(params) == dict:
        pass
    else:
        if params is None:
            params = {}
        elif params.endswith(".json"):
            params = json.load(open(params))
        else:
            db = DB()
            db.load(db_folder)
            job_summary = params
            job = db.get_job_by_summary(params)
            db.close()
            assert job, "Job does not exist : {}".format(params)
            params = job['content']
            assert params is not None
    params["job_id"] = os.getenv("SLURM_JOBID")
    if update_db:
        db = DB()
        db.load(db_folder)
        state = db.get_state_of(job_summary)
        if state != PENDING:
            logger.error("state of the job is not pending, state is : {}".format(state))
            return
        db.modify_state_of(job_summary, RUNNING)
        db.job_update(job_summary, {'slurm_job_id': os.getenv('SLURM_JOBID')})
        db.close()
    if force_w is not None:
        w = int(force_w)
    else:
        w = None
    if force_h is not None:
        h = int(force_h)
    else:
        h = None
    state = 2
    np.random.seed(state)
    logger.info("Loading data...")
    mode = params.get("mode", "random")

    batch_size = params.get("batch_size", 128)
    data_kw = params.get("data_params", {})
    data, w, h, c, nbl, nbc = load_data(
        dataset=dataset, w=w, h=h, include_test=True,
        batch_size=batch_size,
        mode=mode,
        **data_kw)
    # nbl and nbc are just used to show couple of nblxnbc reconstructed
    # vs true samples
    model_params = params.get("model_params", {})
    nb_filters = model_params.get("nb_filters", 128)
    kw_builder = dict(
        nb_filters=nb_filters,
        w=w, h=h, c=c
    )
    if model_name in ("model18", "model20"):
        u = dict(
            nb_layers=2,
            size_filters=6
        )
        kw_builder.update(u)
    elif model_name in ("model19", "model21",):
        u = dict(
            nb_layers=2,
            size_filters=5
        )
        kw_builder.update(u)
    elif model_name in ("model24", "model25", "model26", "model27", "model28", "model29", "model30", "model33"):
        u = dict(
            nb_layers=3,
            size_filters=2**3+2
        )
        kw_builder.update(u)
    elif model_name in ("model34", "model35", "model36", "model37", "model38"):
        u = dict(nb_layers=3, size_filters=2**3+2)
        kw_builder.update(u)
        kw_builder["nb_filters"] = 128
    kw_builder.update(model_params)
    builder = getattr(model, model_name)

    # build the model and return layers dictionary
    logger.info("Building the architecture..")
    layers = builder(**kw_builder)

    info = params.copy()
    def report_event(status):
        #  periodically save the model
        print("Saving the model...")
        save_(layers, builder, kw_builder, "{}/model.pkl".format(prefix), info=info)

    logger.info("Compiling the model...")
    capsule = build_capsule_(layers, data, nbl, nbc,
                             report_event, prefix=prefix,
                             **params)
    info["stats"] = capsule.batch_optimizer.stats
    # info["optimization"] = capsule.batch_optimizer.optim_params
    if mode == 'random':
        dummy = np.zeros((1, c, w, h)).astype(np.float32)
        V = {"X": dummy}
        if "y" in layers:
            dummy_y = np.zeros((1,)).astype(np.int32)
            V.update({"y": dummy_y})
    elif mode == 'minibatch':
        V = {"X": capsule.preprocess(data.X)}
        # y not handled for now
    else:
        raise Exception('not supported mode {}'.format(mode))

    logger.info("Start training!")
    try:
        capsule.fit(**V)
    except KeyboardInterrupt:
        print("keyboard interrupt.")
        pass
    except Exception:
        if update_db:
            db = DB()
            db.load(db_folder)
            db.modify_state_of(job_summary, ERROR)
            db.close()
        raise
    print(capsule.__dict__.keys())
    # save model and report at the end
    capsule.report(capsule.batch_optimizer.stats[-1])
    print("Ok finished training")
    if update_db:
        db = DB()
        db.load(db_folder)
        db.modify_state_of(job_summary, SUCCESS)
        db.close()


def build_capsule_(layers, data, nbl, nbc,
                   report_event=None,
                   prefix="",
                   budget_hours=np.inf,
                   compile_="all",
                   **train_params):
    batch_size = train_params.get("batch_size", 128)
    denoise = train_params.get("denoise", None)
    walkback = train_params.get("walkback", 1)
    autoencoding_loss = train_params.get("autoencoding_loss", "squared_error")

    if denoise is not None:
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        from theano.tensor.shared_randomstreams import RandomStreams as RandomStreamsCpu
        s =  np.random.randint(0, 999999)
        theano_rng = RandomStreams(seed=s)
        theano_rng_cpu = RandomStreamsCpu(seed=s)
    else:
        theano_rng = None
        theano_rng_cpu = None

    if report_event is None:
        def report_event(s):
            pass
    is_predictive = True if "y" in layers else False
    has_factors = True if "factors" in layers else False
    # we only have inputs (X) : no labels
    input_variables = OrderedDict()
    input_variables["X"] = dict(tensor_type=T.tensor4)

    if is_predictive:
        input_variables["y"] = dict(tensor_type=T.ivector)

    # build the convnet layers and the model
    model = InputOutputMapping([layers["input"]], [layers["output"]])

    c, w, h = layers["input"].output_shape[1:]

    def reconstruct(model, X):
        X_rec, = model.get_output(X, deterministic=True)
        return X_rec

    def stoch_reconstruct(model, X):
        X_rec, = model.get_output(X)
        return X_rec

    def predict(model, X):
        return L.get_output(layers["y"], X, deterministic=True)

    def recons_loss(true, pred):
        if autoencoding_loss == "squared_error":
            return ((true - pred) ** 2).sum(axis=(1, 2, 3)).mean()
        elif autoencoding_loss == "cross_entropy":
            pred = theano.tensor.clip(pred, 0.001, 0.999)
            return (T.nnet.binary_crossentropy(pred, true)).sum(axis=(1, 2, 3)).mean()
    def get_recons_loss(model, X):
        Xrec = reconstruct(model, X)
        return recons_loss(X, Xrec)

    functions = {
        "reconstruct": make_function(func=reconstruct, params=["X"]),
        #"get_conv_layers": make_function(func=get_conv, params=["X"]),
        "get_recons_loss": make_function(func=get_recons_loss, params=["X"])
    }
    if is_predictive:
        functions.update({
            "predict": make_function(func=predict, params=["X"])})


    def report(status):
        c, w, h = layers["input"].output_shape[1:]
        ep = status["epoch"]

        rec = capsule.reconstruct
        # save reconstructions
        print('save recons')
        X_orig = data.X[0:nbl * nbc]
        print(X_orig.max())
        X_pred = rec(preprocess(X_orig))
        if layers['input'].output_shape[1] == 3:
            X_orig = X_orig.reshape((X_orig.shape[0], 3, w, h)).transpose((0, 2, 3, 1))
            img_orig = dispims_color(X_orig, border=1)
            X_pred = X_pred.reshape((X_pred.shape[0], 3, w, h)).transpose((0, 2, 3, 1))
            img_pred = dispims_color(X_pred, border=1)
            img = np.concatenate((img_orig, img_pred), axis=1)
        elif layers['input'].output_shape[1] == 1:
            X_orig = (X_orig.reshape((X_orig.shape[0], 1, w, h)) * np.ones((1, 3, 1, 1))).transpose((0, 2, 3, 1))
            img_orig = dispims_color(X_orig, border=1)
            X_pred = (X_pred.reshape((X_pred.shape[0], 1, w, h)) * np.ones((1, 3, 1, 1))).transpose((0, 2, 3, 1))
            img_pred = dispims_color(X_pred, border=1)
            img = np.concatenate((img_orig, img_pred), axis=1)
        imsave("{}/recons/{:08d}.png".format(prefix, ep), img)
        # save features (raw)
        print('save features')
        layer_names = layers.keys()
        for layer_name in layer_names:
            filename = "{}/features/{:08d}-{}.png".format(prefix, ep, layer_name)
            if not hasattr(layers[layer_name], "W"):
                continue
            try:
                W = layers[layer_name].W.get_value().copy()
            except Exception as e:
                print(str(e))
                continue
            if len(W.shape) == 2:
                if W.shape[0] == c * w * h:
                    W = W.T
                if W.shape[1] == c * w * h:
                    W = W.reshape((W.shape[0], c, w , h))
            if 3 in W.shape[0:2]:
                if W.shape[0] == 3:
                    W = W.transpose((1, 2, 3, 0))  # F w h col
                elif W.shape[1] == 3:
                    W = W.transpose((0, 2, 3, 1))  # F w h col
                if W.shape[0] > 512:
                    W = W[0:512]
                img = dispims_color(W, border=1, shape=(11, 11))
                imsave(filename, img)
                #plt.axis('off')
                #plt.imshow(img, interpolation='none')
            elif 1 in W.shape[0:2]:
                W = W.reshape((W.shape[0] * W.shape[1],
                               W.shape[2], W.shape[3]))
                if W.shape[0] > 512:
                    W = W[0:512]
                sz = int(np.sqrt(W.shape[0]))
                print(w, h, W.shape)
                img = tile_raster_images(W, (W.shape[1], W.shape[2]), (sz, sz),
                                         scale_rows_to_unit_interval=True,
                                         output_pixel_vals=True,
                                         tile_spacing=(1, 1))
                imsave(filename, img)
            else:
                continue

        stats = capsule.batch_optimizer.stats

        df = pd.DataFrame(stats)
        df.to_csv("{}/out/stats.csv".format(prefix))
        # learning curve
        epoch = get_stat("epoch", stats)
        if mode == 'random':
            avg_loss = get_stat("avg_loss_train_fix", stats)
            pd.Series(avg_loss).to_csv("{}/out/avg_loss.csv".format(prefix))

        loss = get_stat("loss_train", stats)
        pd.Series(loss).to_csv("{}/out/loss.csv".format(prefix))
        if is_predictive:
            acc = [s["acc_test"] for s in stats if "acc_test" in s]
            pd.Series(acc).to_csv("{}/out/acc.sv".format(prefix))

        fig = plt.figure()
        if mode == 'random':
            plt.plot(epoch, avg_loss, label="avg_loss")
        plt.plot(epoch, loss, label="loss")
        plt.xlabel("x")
        plt.ylabel("loss")
        plt.savefig("{}/out/avg_loss_train.png".format(prefix))
        plt.legend()
        plt.close(fig)

        report_event(status)

    begin = datetime.now()
    budget_sec = budget_hours * 3600
    # called each epoch for monitoring

    def update_status(self, status):
        status['duration'] =  (datetime.now() - self.last_checkpoint).total_seconds()
        self.last_checkpoint = datetime.now()
        t = status["epoch"]
        cur_lr = lr.get_value()

        if lr_decay_method == "exp":
            new_lr = cur_lr * (1 - lr_decay)
        elif lr_decay_method == "lin":
            new_lr = initial_lr / (1 + t)
        elif lr_decay_method == "sqrt":
            new_lr = initial_lr / np.sqrt(1 + t)
        else:
            new_lr = cur_lr

        new_lr = np.array(new_lr, dtype="float32")
        lr.set_value(new_lr)

        if mode == 'random':
            B = 0.9  # moving window param for estimating loss_train
            loss = "loss_train"
            loss_avg = "avg_{}".format(loss)
            if len(self.stats) == 1:
                last_avg = 0
            else:
                last_avg = self.stats[-2][loss_avg]
            status["avg_loss_train"] = B * last_avg + (1 - B) * status[loss]
            fix = 1 - B ** (1 + t)
            status["avg_loss_train_fix"] = status["avg_loss_train"] / fix
        
        N = 200
        if is_predictive and hasattr(data, "test") and t % N == 0:
            preds = []
            for batch in iterate_minibatches(data.test.X.shape[0], batchsize=1000):
                pred = capsule.predict(capsule.preprocess(data.test.X[batch])).argmax(axis=1) == data.test.y[batch]
                preds.append(pred)
            acc = (np.concatenate(preds, axis=0)).mean()
            status["acc_test"] = acc

        if mode == "minibatch":
            N = 5
        elif mode == 'random':
            N = 1000
        else:
            raise Exception('wtf how come mode is not valid and it happens here')
        # each N epochs save reconstructions
        # and how features look like
        if t % N == 0:
            for name in ("train", "test"):
                if hasattr(data, name):
                    dt = getattr(data, name)
                    rec_errors = []
                    for batch in iterate_minibatches(dt.X.shape[0], batchsize=1000):
                        rec_error = capsule.get_recons_loss(capsule.preprocess(dt.X[batch]))
                        rec_errors.append(rec_error)
                    rec_error_mean = np.mean(rec_errors)
                    status["{}_recons_error".format(name)] = rec_error_mean
        if t % N == 0:
            report(status)
        if (datetime.now() - begin).total_seconds() >= budget_sec:
            logger.info("Budget finished.quit.")
            raise KeyboardInterrupt("Budget finished.quit.")
        return status

    # Initialize the optimization algorithm
    mode = train_params.get("mode", "random")

    optim_params_default = dict(
        lr_decay_method="none",
        initial_lr=0.1,
        lr_decay=0,
        algo="adadelta",
        momentum=0.9,
        beta1=0.95,
        beta2=0.95,
        epsilon=1e-8,
        max_nb_epochs=100000 if mode=='random' else 213, # approx 213 nb of epochs corresponds to nb of epochs to do 100000 with batchsize 128 on training data of size 60000
        patience_stat='avg_loss_train_fix' if mode == 'random' else 'loss_train',
        patience_nb_epochs=800 if mode == 'random' else 20,
        min_nb_epochs=5000 if mode == 'random' else 10,
        batch_size=batch_size,
    )
    optim_params = optim_params_default.copy()
    optim_params.update(train_params.get("optimization", {}))
    lr_decay_method = optim_params["lr_decay_method"]
    initial_lr = optim_params["initial_lr"]
    lr_decay = optim_params["lr_decay"]
    lr = theano.shared(np.array(initial_lr, dtype=np.float32))
    algos = {"adam": updates.adam, "adadelta": updates.adadelta, "momentum": updates.momentum}
    algo = algos[optim_params["algo"]]
    params = {"learning_rate": lr}

    if algo in (updates.momentum, updates.nesterov_momentum):
        params["momentum"] = optim_params["momentum"]
    if algo == updates.adam:
        params["beta1"] = optim_params["beta1"]
        params["beta2"] = optim_params["beta2"]
        params["epsilon"] = optim_params["epsilon"]
    optim = (algo, params)
    batch_optimizer = make_batch_optimizer(
        update_status,
        max_nb_epochs=optim_params["max_nb_epochs"],
        optimization_procedure=optim,
        patience_stat=optim_params["patience_stat"],
        patience_nb_epochs=optim_params["patience_nb_epochs"],
        min_nb_epochs=optim_params["min_nb_epochs"],
        batch_size=batch_size,
        whole_dataset_in_device=train_params.get('whole_dataset_in_device', False),
        verbose=1)
    batch_optimizer.lr = lr
    batch_optimizer.optim_params = optim_params
    batch_optimizer.last_checkpoint = datetime.now()
    
    def noisify(X):
        pr = float(denoise)
        if pr == 0:
            return X
        noise = train_params.get("noise", "zero_masking")
        if noise == "salt_and_pepper":
            Xtilde = salt_and_pepper(X, corruption_level=pr, rng=theano_rng, backend='theano')
        elif noise == "zero_masking":
            Xtilde = zero_masking(X, corruption_level=pr, rng=theano_rng)
        elif noise == "superpose":
            perm = theano_rng_cpu.permutation(n=X.shape[0])
            print(perm.ndim)
            Xtilde = X * pr + X[perm] * (1 - pr)
        return Xtilde

    def stoch_reconstruct_and_sample(model, X):
        Xrec = stoch_reconstruct(model, X)
        return bernoulli_sample(Xrec, theano_rng)
    
    def loss_function(model, tensors):
        X = tensors["X"]
        if denoise is not None:
            # for backward compatibility, but walkback_jump should not be used
            # anymore, use walkback_mode only
            if train_params.get("walkback_jump", True) is True:
                walkback_mode = train_params.get("walkback_mode", "mine2")
            else:
                walkback_mode = 'mine'

            if walkback_mode == "mine2":
                recons = 0
                Xtilde = noisify(X)
                X_pred = stoch_reconstruct(model, Xtilde)
                for i in range(walkback):
                    recons += recons_loss(X, X_pred)
                    X_pred = stoch_reconstruct(model, X_pred)
                loss = recons
            elif walkback_mode == "bengio_without_sampling":
                recons = 0
                Xcur = X
                for i in range(walkback):
                    Xcur = noisify(Xcur)
                    Xcur = stoch_reconstruct(model, Xcur)
                    recons += recons_loss(X, Xcur)
                loss = recons
            elif walkback_mode == "bengio":
                recons = 0
                Xcur = X
                for i in range(walkback):
                    Xcur = noisify(Xcur)
                    recons += recons_loss(X, stoch_reconstruct(model, Xcur))
                    Xcur = stoch_reconstruct_and_sample(model, Xcur)
                loss = recons
            elif walkback_mode == "mine":
                recons = 0
                Xcur = X
                for i in range(walkback):
                    Xcur_tilde = noisify(Xcur)
                    recons += recons_loss(Xcur, stoch_reconstruct(model, Xcur_tilde))
                    Xcur = Xcur_tilde
                loss = recons
            else:
                raise Exception('no valid walkback mode')

            updates =  [(v, u) for v, u, _, _ in theano_rng.updates()]
            updates.extend([(v, u) for v, u in theano_rng_cpu.updates()])
        else:
            X_pred = stoch_reconstruct(model, X)
            recons = recons_loss(X, X_pred)
            loss = recons
            updates = []

        if is_predictive:
            y_pred = predict(model, X)
            classif = -T.log(y_pred[T.arange(y_pred.shape[0]), tensors["y"]]).mean()
            lambda_ = train_params.get("lambda_", 10.)
            loss += lambda_ * classif
        if has_factors:
            assert is_predictive
            y_pred = predict(model, X)
            for layer_name, layer in layers.items():
                if layer_name.startswith("factor") and type(layer) == L.DenseLayer:
                    mu = train_params.get("mu", 10.)
                    loss += mu * cross_correlation(L.get_output(layer, X), y_pred)
        print(train_params)
        if train_params.get("contractive", False) is True:
            from lasagne import nonlinearities
            contcoef = train_params.get("contractive_coef", 0.1)
            contlayers = train_params.get("contractive_layers", layers.keys())
            for layername in contlayers:
                layer = layers[layername]
                if hasattr(layer, "W"):
                    hid = L.get_output(layer, X)
                    print("contracting...")
                    if layer.nonlinearity == nonlinearities.sigmoid:
                        hid = hid.dimshuffle(0, 'x', 1)
                        W = layer.W.dimshuffle('x', 0, 1)
                        if train_params.get("marginalized", False) is True:
                            coefs = 0.5 * (W**2).sum(axis=(0, 1)) #equation 7 in marginalized denoising auto-encoders
                            coefs = coefs.dimshuffle('x', 0)
                        else:
                            coefs = 1
                        J = ((hid * (1 - hid) * W)**2)
                        loss += contcoef * (coefs * J).sum() / hid.shape[0]
                    else:
                        print("Unknown nonlinearity : {}, do not contract".format(layer.nonlinearity))
        if train_params.get("ladder", False) is True:
            lambda_ = train_params.get("lambda", 0.1)
            for layername in layers.keys():
                if layername.startswith("enc"):
                    enc = layers[layername]
                    dec = layers[layername.replace("enc", "dec")]
                    print(enc.name, dec.name, enc.output_shape, dec.output_shape)
                    rec_error = ((L.get_output(enc, X) - L.get_output(dec, X))**2).sum(axis=1).mean()
                    loss += lambda_ * rec_error
        return loss, updates

    def transform(batch_index, batch_slice, tensors):
        data.load()
        t = OrderedDict()
        t["X"] = preprocess(data.X)
        if is_predictive:
            t["y"] = data.y
        return t

    def preprocess(X):
        X = X.reshape((X.shape[0], c, w, h))
        if train_params.get("binarize_thresh", None) is not None:
            thresh = train_params.get("binarize_thresh", 0.5)
            return X > thresh
        else:
            return X

    mode = train_params.get("mode", "random")
    if batch_optimizer.whole_dataset_in_device is True:
        assert mode == 'minibatch'
    if mode == "random":
        batch_iterator = build_batch_iterator(transform)
    elif mode == "minibatch":
        batch_iterator = None
    else:
        raise Exception('Unknown mode : {}'.format(mode))
    # put all together
    capsule = Capsule(
        input_variables,
        model,
        loss_function,
        functions=functions,
        batch_optimizer=batch_optimizer,
        batch_iterator=batch_iterator,
        rng=theano_rng
    )
    capsule.layers = layers
    capsule.preprocess = preprocess
    if compile_ == "all":
        mode = train_params.get("mode", "random")
        if mode == "random":
            dummy = np.zeros((1, c, w, h)).astype(np.float32)
            dummyvars = {"X": dummy}
            if is_predictive:
                dummyvars.update({"y": dummy})
            capsule._build(dummyvars)
        elif mode == "minibatch":
            V = {"X": capsule.preprocess(data.X)}
            capsule._build(V)
    elif compile_ == "functions_only":
        capsule._build_functions()
    elif compile_ == "none":
        pass
    capsule.report = report
    return capsule


@task
def check(filename="out.pkl",
          what="filters",
          dataset="fonts",
          prefix="",
          force_w=None,
          force_h=None,
          params=None,
          folder=None,
          update_db=None,
          batch_size=128,
          kw_load_data=None):
    import json
    import traceback
    import analyze
    logger.info("Loading data...")
    if force_w is not None:
        w = force_w
    else:
        w = None
    if force_h is not None:
        h = force_h
    else:
        h = None
    if kw_load_data is None:
        kw_load_data = {}
    data, w, h, c, nbl, nbc = load_data(dataset=dataset, w=w, h=h, batch_size=batch_size, **kw_load_data)
    logger.info("Loading the model...")
    layers, model_params = load_(filename, w=w, h=h)
    logger.info("Compiling the model...")
    capsule = build_capsule_(
            layers, data, nbl, nbc,
            prefix=prefix,
            compile_="functions_only",
            **model_params)
    if type(params) == list:
        pass
    elif type(params) == dict:
        params = [params]
    elif params is None:
            params = [{}]
    elif params.endswith(".json"):
        params = [json.load(open(params))]
    else:
        job_summary = params
        db = DB()
        db.load(db_folder)
        job = db.get_job_by_summary(params)
        db.close()
        assert job, "Job does not exist : {}".format(params)
        params = job['content']['check']['params']
        assert params is not None
        print(params)
        params = [params]

    if update_db:
        db = DB()
        db.load(db_folder)
        state = db.get_state_of(job_summary)
        if state != PENDING:
            logger.error("state of the job is not pending, state is : {}".format(state))
            return
        db.modify_state_of(job_summary, RUNNING)
        db.job_update(job_summary, {'slurm_job_id': os.getenv('SLURM_JOBID')})
        db.close()

    assert hasattr(analyze, what)
    func = getattr(analyze, what)
    ret = None
    for p in params:
        try:
            state = p.get("seed", 2)
            np.random.seed(state)
            p["seed"] = state
            print(p)
            ret = func(capsule, data, layers, w, h, c, folder, **p)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            #traceback.print_tb(exc_traceback, limit=4, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=5, file=sys.stdout)
            if update_db:
                db = DB()
                db.load(db_folder)
                db.modify_state_of(job_summary, ERROR)
                db.close()
                return None
            else:
                return None

    if update_db:
        db = DB()
        db.load(db_folder)
        db.modify_state_of(job_summary, SUCCESS)
        db.close()
    return ret


def save_(layers, builder, kw_builder, filename, info=None):
    if info is None:
        info = {}
    with open(filename, "w") as fd:
        values = L.get_all_param_values(layers["output"])
        data = {"values": values, "builder": builder, "kw_builder": kw_builder, "info": info}
        dill.dump(data, fd)


def load_(filename, **kwargs):
    with open(filename, "r") as fd:
        data = dill.load(fd)
        builder, kw_builder, values = (
            data["builder"], data["kw_builder"], data["values"])
        kw_builder.update(**kwargs)
    layers = builder(**kw_builder)
    L.set_all_param_values(layers["output"], values)
    return layers, data.get("info", {})
