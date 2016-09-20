import logging
import sys

import theano
import theano.tensor as T
from lasagne import layers as L

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def squared_error(true, pred):
    return ((true - pred) ** 2).sum(axis=(1, 2, 3)).mean()

def mean_squared_error(true, pred):
    return ((true - pred) ** 2).mean()

def cross_entropy(true, pred):
    pred = theano.tensor.clip(pred, 0.001, 0.999)
    return (T.nnet.binary_crossentropy(pred, true)).sum(axis=(1, 2, 3)).mean()

def vae_kl_div(z_mu, z_log_sigma):
    return -0.5 * (1 + 2*z_log_sigma - T.sqr(z_mu) - T.exp(2 * z_log_sigma))

def vae_loss_binary(X, mu, z_mu, z_log_sigma):
    eps = 10e-8
    mu = theano.tensor.clip(mu, eps, 1 - eps)  # like keras
    binary_ll = (T.nnet.binary_crossentropy(mu, X)).sum(axis=1).mean()
    kl_div = vae_kl_div(z_mu, z_log_sigma).sum(axis=1).mean()
    return binary_ll + kl_div

def vae_loss_real(X, mu, log_sigma, z_mu, z_log_sigma):
    gaussian_ll = gaussian_log_likelihood(X, mu, log_sigma).sum(axis=1).mean()
    kl_div = vae_kl_div(z_mu, z_log_sigma).sum(axis=1).mean()
    return (gaussian_ll + kl_div)
