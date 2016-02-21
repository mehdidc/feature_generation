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
from lasagnekit.misc.plot_weights import grid_plot, dispims_color
from lasagnekit.easy import get_stat
import numpy as np
from data import load_data
from model import *  # for dill
import logging

sys.setrecursionlimit(10000)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@task
def train(dataset="digits", prefix="", model_name="model8", force_w=None, force_h=None):
    if force_w is not None:
        w = force_w
    else:
        w = None

    if force_h is not None:
        h = force_h
    else:
        h = None
    state = 2
    np.random.seed(state)
    logger.info("Loading data...")
    data, w, h, c, nbl, nbc = load_data(dataset=dataset, w=w, h=h)
    # nbl and nbc are just used to show couple of nblxnbc reconstructed
    # vs true samples

    nb_filters = 64
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
    elif model_name in ("model24", "model25", "model26", "model27", "model28", "model29", "model30"):
        u = dict(
            nb_layers=3,
            size_filters=2**3+2
        )
        kw_builder.update(u)

    builder = getattr(model, model_name)

    # build the model and return layers dictionary
    logger.info("Building the architecture..")
    layers = builder(**kw_builder)

    def report_event(status):
        #  periodically save the model
        print("Saving the model...")
        save_(layers, builder, kw_builder, "{}/model.pkl".format(prefix))

    logger.info("Compiling the model...")
    capsule = build_capsule_(layers, data, nbl, nbc,
                             report_event, prefix=prefix)
    dummy = np.zeros((1, c, w, h)).astype(np.float32)
    logger.info("Start training!")
    try:
        capsule.fit(X=dummy)
    except KeyboardInterrupt:
        print("keyboard interrupt.")
        pass
    print(capsule.__dict__.keys())
    # save model and report at the end
    capsule.report(capsule.batch_optimizer.stats[-1])


def build_capsule_(layers, data, nbl, nbc,
                   report_event=None,
                   prefix="",
                   compile_="all"):  # all/functions only/none
    if report_event is None:
        def report_event(s):
            pass
    # we only have inputs (X) : no labels
    input_variables = OrderedDict()
    input_variables["X"] = dict(tensor_type=T.tensor4)
    # build the convnet layers and the model
    model = InputOutputMapping([layers["input"]], [layers["output"]])

    c, w, h = layers["input"].output_shape[1:]

    def reconstruct(model, X):
        y, = model.get_output(X)
        return y

    def is_conv_layer(name):
        return name.startswith("conv1")
    conv_layer_names = filter(is_conv_layer, layers.keys())
    conv_layer_names += ["output"]
    conv_layers = map(lambda name: layers[name], conv_layer_names)

    def get_conv(model, X):
        out = []
        for layer in conv_layers:
            out.append(L.get_output(layer, X))
        return out

    functions = {
        "reconstruct": make_function(func=reconstruct, params=["X"]),
        "get_conv_layers": make_function(func=get_conv, params=["X"])
    }

    def report(status):
        ep = status["epoch"]
        X_pred = capsule.reconstruct(preprocess(data.X))
        # save reconstructions
        k = 1
        idx = 0
        fig = plt.figure()
        for l in range(nbl):
            for c in range(nbc):
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
            fig = plt.figure()
            fig.patch.set_facecolor('gray')
            if not hasattr(layers[layer_name], "W"):
                continue
            W = layers[layer_name].W.get_value().copy()
            if 3 in W.shape[0:2]:
                if W.shape[0] == 3:
                    W = W.transpose((1, 2, 3, 0))  # F w h col
                elif W.shape[1] == 3:
                    W = W.transpose((0, 2, 3, 1))  # F w h col
                img = dispims_color(W, border=1, shape=(11, 11))
                plt.axis('off')
                plt.imshow(img, interpolation='none')
            elif 1 in W.shape[0:2]:
                W = layers[layer_name].W.get_value().copy()
                W = W.reshape((W.shape[0] * W.shape[1],
                               W.shape[2], W.shape[3]))
                opt = dict(cmap='gray', interpolation='none')
                grid_plot(W, imshow_options=opt, fig=fig)
            else:
                continue
            plt.savefig("{}/features/{}-{}.png".format(prefix, ep, layer_name),
                        facecolor=fig.get_facecolor(), transparent=True)
            plt.close(fig)

        # learning curve
        stats = capsule.batch_optimizer.stats
        epoch = get_stat("epoch", stats)
        avg_loss = get_stat("avg_loss_train_fix", stats)
        loss = get_stat("loss_train", stats)
        pd.Series(avg_loss).to_csv("{}/out/avg_loss.csv".format(prefix))
        pd.Series(loss).to_csv("{}/out/loss.csv".format(prefix))
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

        N = 1000
        # each N epochs save reconstructions
        # and how features look like
        if t % N == 0:
            report(status)

        return status

    # Initialize the optimization algorithm
    lr_decay_method = "none"
    initial_lr = 0.1
    lr_decay = 0
    lr = theano.shared(np.array(initial_lr, dtype=np.float32))
    # algo = updates.momentum
    algo = updates.adadelta
    # algo = updates.adam
    params = {"learning_rate": lr}

    if algo in (updates.momentum, updates.nesterov_momentum):
        momentum = 0.9
        params["momentum"] = momentum
    if algo == updates.adam:
        params["beta1"] = 0.9
    optim = (algo, params)
    batch_size = nbl * nbc
    batch_optimizer = make_batch_optimizer(
        update_status,
        max_nb_epochs=100000,
        optimization_procedure=optim,
        patience_stat='avg_loss_train_fix',
        patience_nb_epochs=800,
        min_nb_epochs=25000,
        batch_size=batch_size,
        verbose=1)
    batch_optimizer.lr = lr

    def loss_function(model, tensors):
        X = tensors["X"]
        X_pred = reconstruct(model, X)
        return ((X - X_pred) ** 2).sum(axis=(1, 2, 3)).mean()

    def transform(batch_index, batch_slice, tensors):
        data.load()
        t = OrderedDict()
        t["X"] = preprocess(data.X)
        return t

    def preprocess(X):
        return X.reshape((X.shape[0], c, w, h))

    batch_iterator = build_batch_iterator(transform)

    # put all together
    capsule = Capsule(
        input_variables,
        model,
        loss_function,
        functions=functions,
        batch_optimizer=batch_optimizer,
        batch_iterator=batch_iterator
    )
    capsule.layers = layers
    capsule.preprocess = preprocess
    dummy = np.zeros((1, c, w, h)).astype(np.float32)
    if compile_ == "all":
        capsule._build({"X": dummy})
    elif compile_ == "functions_only":
        capsule._build_functions()
    elif compile__ == "none":
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
          params=None):
    import json
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
    data, w, h, c, nbl, nbc = load_data(dataset=dataset, w=w, h=h)
    logger.info("Loading the model...")
    layers = load_(filename, w=w, h=h)
    logger.info("Compiling the model...")
    capsule = build_capsule_(
            layers, data, nbl, nbc,
            prefix=prefix,
            compile_="functions_only")
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
        #try:
        state = p.get("seed", 2)
        np.random.seed(state)
        p["seed"] = state
        ret=func(capsule, data, layers, w, h, c, **p)
        #except Exception as e:
        #    print(str(e))
    return ret

def save_(layers, builder, kw_builder, filename):
    with open(filename, "w") as fd:
        values = L.get_all_param_values(layers["output"])
        data = {"values": values, "builder": builder, "kw_builder": kw_builder}
        dill.dump(data, fd)


def load_(filename, **kwargs):
    with open(filename, "r") as fd:
        data = dill.load(fd)
        builder, kw_builder, values = (
            data["builder"], data["kw_builder"], data["values"])
        kw_builder.update(**kwargs)
    layers = builder(**kw_builder)
    L.set_all_param_values(layers["output"], values)
    return layers
