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

from lasagne import updates
from lasagnekit.easy import (
    make_batch_optimizer, InputOutputMapping,
    build_batch_iterator)
from lasagne import layers as L
from lasagnekit.nnet.capsule import Capsule, make_function
from lasagnekit.misc.plot_weights import grid_plot, dispims_color, tile_raster_images
from lasagnekit.easy import get_stat, iterate_minibatches
import numpy as np
from helpers import salt_and_pepper, zero_masking
from data import load_data
from model import *  # for dill
from skimage.io import imsave
import logging

sys.setrecursionlimit(10000)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)

@task
def train(dataset="digits", prefix="",
          model_name="model8",
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
        else:
            params = json.load(open(params))
    params["job_id"] = os.getenv("SLURM_JOBID")
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
    data, w, h, c, nbl, nbc = load_data(
        dataset=dataset, w=w, h=h, include_test=True,
        batch_size=batch_size,
        mode=mode)
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

    logger.info("Start training!")
    try:
        capsule.fit(**V)
    except KeyboardInterrupt:
        print("keyboard interrupt.")
        pass
    print(capsule.__dict__.keys())
    # save model and report at the end
    capsule.report(capsule.batch_optimizer.stats[-1])


def build_capsule_(layers, data, nbl, nbc,
                   report_event=None,
                   prefix="",
                   compile_="all",
                   **train_params):
    batch_size = train_params.get("batch_size", 128)
    denoise = train_params.get("denoise", None)
    walkback = train_params.get("walkback", 1)
    autoencoding_loss = train_params.get("autoencoding_loss", "squared_error")

    if denoise is not None:
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        theano_rng = RandomStreams(seed=np.random.randint(0, 999999))
    else:
        theano_rng = None

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

    def is_conv_layer(name):
        return name.startswith("conv1")
    conv_layer_names = filter(is_conv_layer, layers.keys())
    conv_layer_names += ["output"]
    conv_layers = map(lambda name: layers[name], conv_layer_names)

    def get_conv(model, X):
        out = []
        for layer in conv_layers:
            out.append(L.get_output(layer, X, determnistic=True))
        return out

    functions = {
        "reconstruct": make_function(func=reconstruct, params=["X"]),
        "get_conv_layers": make_function(func=get_conv, params=["X"])
    }
    if is_predictive:
        functions.update({
            "predict": make_function(func=predict, params=["X"])})


    def report(status):
        c, w, h = layers["input"].output_shape[1:]
        ep = status["epoch"]
        X_pred = capsule.reconstruct(preprocess(data.X))
        # save reconstructions
        k = 1
        idx = 0
        fig = plt.figure(figsize=(10, 10))
        for row in range(nbl):
            for col in range(nbc):
                plt.subplot(nbl, nbc * 2, k)
                plt.axis('off')
                if idx >= len(data.X):
                    break
                if layers['input'].output_shape[1] != 3:
                    plt.imshow(1 - data.X[idx].reshape((w, h)),
                               cmap="gray", interpolation='none')
                else:
                    img = data.X[idx].reshape((3, w, h))
                    img = img.transpose((1, 2, 0))
                    plt.imshow(img, interpolation='none')
                k += 1
                plt.subplot(nbl, nbc * 2, k)
                plt.axis('off')
                if layers['input'].output_shape[1] != 3:
                    plt.imshow(1 - X_pred[idx][0],
                               cmap="gray", interpolation='none')
                else:
                    img = X_pred[idx].reshape((3, w, h))
                    img = img.transpose((1, 2, 0))
                    plt.imshow(img, interpolation='none')

                k += 1
                idx += 1
        plt.savefig("{}/recons/{}.png".format(prefix, ep))
        plt.close(fig)

        # save features (raw)
        layer_names = layers.keys()
        for layer_name in layer_names:

            filename = "{}/features/{}-{}.png".format(prefix, ep, layer_name)

            fig = plt.figure()
            fig.patch.set_facecolor('gray')
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
                img = dispims_color(W, border=1, shape=(11, 11))
                imsave(filename, img)
                #plt.axis('off')
                #plt.imshow(img, interpolation='none')
            elif 1 in W.shape[0:2]:
                W = W.reshape((W.shape[0] * W.shape[1],
                               W.shape[2], W.shape[3]))
                sz = int(np.sqrt(W.shape[0]))
                print(w, h, W.shape)
                img = tile_raster_images(W, (W.shape[1], W.shape[2]), (sz, sz),
                                         scale_rows_to_unit_interval=True,
                                         output_pixel_vals=True,
                                         tile_spacing=(1, 1))
                #plt.axis('off')
                #plt.imshow(img, cmap='gray', interpolation='none')
                imsave(filename, img)
                #opt = dict(cmap='gray', interpolation='none')
                    #grid_plot(W, imshow_options=opt, fig=fig)
                    #plt.axis('off')
                    #plt.savefig(filename, facecolor=fig.get_facecolor(), transparent=True)
            else:
                continue
            plt.close(fig)

        # learning curve
        stats = capsule.batch_optimizer.stats
        epoch = get_stat("epoch", stats)
        avg_loss = get_stat("avg_loss_train_fix", stats)
        loss = get_stat("loss_train", stats)
        pd.Series(avg_loss).to_csv("{}/out/avg_loss.csv".format(prefix))
        pd.Series(loss).to_csv("{}/out/loss.csv".format(prefix))
        if is_predictive:
            acc = [s["acc_test"] for s in stats if "acc_test" in s]
            pd.Series(acc).to_csv("{}/out/acc.sv".format(prefix))
        fig = plt.figure()
        plt.plot(epoch, avg_loss, label="avg_loss")
        plt.plot(epoch, loss, label="loss")
        plt.xlabel("x")
        plt.ylabel("loss")
        plt.savefig("{}/out/avg_loss_train.png".format(prefix))
        plt.legend()
        plt.close(fig)

        report_event(status)

    # called each epoch for monitoring
    def update_status(self, status):
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

        N = 1000
        if mode == "minibatch":
            t = (t * data.X.shape[0] / batch_size)
            next_ = (t + batch_size) / N
            next_ = next_ * N
            if next_ >= (t + batch_size):
                next_ += N
            if next_ >= t and next_ <= t + batch_size:
                t = next_
        # each N epochs save reconstructions
        # and how features look like
        if t % N == 0:
            report(status)

        return status

    # Initialize the optimization algorithm
    optim_params_default = dict(
        lr_decay_method="none",
        initial_lr=0.1,
        lr_decay=0,
        algo="adadelta",
        momentum=0.9,
        beta1=0.95,
        beta2=0.95,
        epsilon=1e-8,
        max_nb_epochs=100000,
        patience_stat='avg_loss_train_fix',
        patience_nb_epochs=800,
        min_nb_epochs=250000,
        batch_size=batch_size,
    )
    optim_params = optim_params_default.copy()
    optim_params.update(train_params.get("optimization", {}))
    lr_decay_method = optim_params["lr_decay_method"]
    initial_lr = optim_params["initial_lr"]
    lr_decay = optim_params["lr_decay"]
    lr = theano.shared(np.array(initial_lr, dtype=np.float32))
    algos = {"adam": updates.adam, "adadelta": updates.adadelta}
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
        verbose=1)
    batch_optimizer.lr = lr
    batch_optimizer.optim_params = optim_params

    def loss_function(model, tensors):
        X = tensors["X"]
        if denoise is not None:
            noise = train_params.get("noise", "salt_and_pepper")
            def noisify(X):
                if noise == "salt_and_pepper":
                    Xtilde = salt_and_pepper(X, corruption_level=float(denoise), rng=theano_rng, backend='theano')
                elif noise == "zero_masking":
                    Xtilde = zero_masking(X, corruption_level=float(denoise), rng=theano_rng)
                return Xtilde

            Xtilde = noisify(X)
            X_pred = stoch_reconstruct(model, Xtilde)
            updates =  [(v, u) for v, u, _, _ in theano_rng.updates()]
        else:
            X_pred = stoch_reconstruct(model, X)
            updates = []

        def recons_loss(true, pred):
            if autoencoding_loss == "squared_error":
                return ((true - pred) ** 2).sum(axis=(1, 2, 3)).mean()
            elif autoencoding_loss == "cross_entropy":
                return (true * T.log(pred) + (1 - true) * T.log(1 - pred)).sum(axis=(1, 2, 3)).mean()

        if train_params.get("walkback_jump", True) is True:
            recons = 0
            for i in range(walkback):
                recons += recons_loss(X, X_pred)
                X_pred = stoch_reconstruct(model, X_pred)
            loss = recons
        else:
            recons = 0
            Xcur = X
            for i in range(walkback):
                Xcur_tilde = noisify(Xcur)
                recons += recons_loss(Xcur, stoch_reconstruct(model, Xcur_tilde))
                Xcur = Xcur_tilde
            loss = recons
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
                            coefs = (W**2).sum(axis=(0, 1))
                            coefs = coefs.dimshuffle('x', 0)
                        else:
                            coefs = 1
                        loss += contcoef * (coefs * ((hid * (1 - hid) * W)**2)).sum() / hid.shape[0]
                    else:
                        raise Exception("unknown nonlinearity")
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

    if train_params.get("mode", "random") == "random":
        batch_iterator = build_batch_iterator(transform)
    else:
        #mode minibatch
        batch_iterator = None
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
    else:
        if params is None:
            params = [{}]
        else:
            params = [json.load(open(params))]

    assert hasattr(analyze, what)
    func = getattr(analyze, what)

    for p in params:
        try:
            state = p.get("seed", 2)
            np.random.seed(state)
            p["seed"] = state
            ret = func(capsule, data, layers, w, h, c, **p)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            #traceback.print_tb(exc_traceback, limit=4, file=sys.stdout)
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=5, file=sys.stdout)
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
