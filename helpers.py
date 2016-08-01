import theano.tensor as T
import numpy as np
import theano
import os
from lasagnekit.easy import iterate_minibatches
import lasagne

def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)



def wta_spatial(X):
    # From http://arxiv.org/pdf/1409.2752v2.pdf
    # Introduce sparsity for each channel/feature map, for each
    # feature map make all activations zero except the max
    # This introduces a sparsity level dependent on the number
    # of feature maps,
    # for instance if we have 10 feature maps, the sparsity level
    # is 1/10 = 10%
    mask = (equals_(X, T.max(X, axis=(2, 3), keepdims=True))) * 1
    return X * mask


def wta_k_spatial(nb=1):

    def apply_(X):
        shape = X.shape
        X_ = X.reshape((X.shape[0] * X.shape[1], X.shape[2] * X.shape[3]))
        idx = T.argsort(X_, axis=1)[:, X_.shape[1] - nb]
        val = X_[T.arange(X_.shape[0]), idx]
        mask = X_ >= val.dimshuffle(0, 'x')
        X_ = X_ * mask
        X_ = X_.reshape(shape)
        return X_
    return apply_


def wta_lifetime(percent):

    def apply_(X):
        X_max = X.max(axis=(2, 3), keepdims=True)  # (B, F, 1, 1)
        idx = (1 - percent) * X.shape[0] - 1
        mask = T.argsort(X_max, axis=0) >= idx  # (B, F, 1, 1)
        return X * mask
    return apply_


def wta_fc_lifetime(percent):
    def apply_(X):
        idx = ((1 - percent) * X.shape[0] - 1)
        mask = T.argsort(X, axis=0) >= idx  # (B, F)
        return X * mask
    return apply_

def wta_fc_sparse(percent):
    def apply_(X):
        idx = ((1 - percent) * X.shape[0] - 1)
        mask = T.argsort(X, axis=1) >= idx  # (B, F)
        return X * mask
    return apply_

def wta_channel(X):
    mask = equals_(X, T.max(X, axis=1, keepdims=True)) * 1
    return X * mask


def wta_channel_strided(stride=2):

    def apply_(X):
        B, F = X.shape[0:2]
        w, h = X.shape[2:]
        X_ = X.reshape((B, F, w / stride, stride, h / stride, stride))
        mask = equals_(X_, X_.max(axis=(1, 3, 5), keepdims=True)) * 1
        mask = mask.reshape(X.shape)
        return X * mask
    return apply_


def equals_(x, y, eps=1e-8):
    return T.abs_(x - y) <= eps

def cross_correlation(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return 0.5 * ((((a.dimshuffle(0, 'x', 1) * b.dimshuffle(0, 1, 'x'))).mean(axis=0))**2).sum()


def salt_and_pepper(x, rng=np.random, backend='theano', corruption_level=1.5):
    if backend == 'theano':
        a = rng.binomial(
            size=x.shape,
            p=(1 - corruption_level),
            dtype=theano.config.floatX
        )
    else:
        a = rng.uniform(size=x.shape) <= (1 - corruption_level)
    if backend == 'theano':
        b = rng.binomial(
            size=x.shape,
            p=0.5,
            dtype=theano.config.floatX
        )
    else:
        b = rng.uniform(size=x.shape) <=  0.5

    if backend == 'theano':
        c = T.eq(a, 0) * b
    else:
        c = (a==0) * b
    return x * a + c

def zero_masking(x, rng, corruption_level=0.5):
    a = rng.binomial(
        size=x.shape,
        p=(1 - corruption_level),
        dtype=theano.config.floatX
    )
    return x * a

def bernoulli_sample(x, rng):
    xs = rng.uniform(size=x.shape) <= x
    return xs

def zero_mask(x, rng, corruption_level=0.5):
    a = rng.binomial(
        size=x.shape,
        p=(1 - corruption_level),
        dtype=theano.config.floatX
    )
    return a


def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


def minibatcher(fn, batchsize=1000):
    def f(X):
        results = []
        for sl in iterate_minibatches(len(X), batchsize):
            results.append(fn(X[sl]))
        return np.concatenate(results, axis=0)
    return f


class MultiSubSampled(object):

    def __init__(self, dataset, nb, random_state=2):
        self.dataset = dataset
        self.nb = nb
        self.rng = np.random.RandomState(random_state)

    def load(self):
        self.dataset.load()
        indices_ax0 = self.rng.randint(0, self.dataset.X.shape[0], size=self.nb)
        indices_ax1 = self.rng.randint(0, self.dataset.X.shape[1], size=self.nb)
        self.X = self.dataset.X[indices_ax0, indices_ax1, :, :]
        if hasattr(self.dataset, "img_dim"):
            self.img_dim = self.dataset.img_dim
        if hasattr(self.dataset, "output_dim"):
            self.output_dim = self.dataset.output_dim


class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 W=lasagne.init.Orthogonal(),
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(W,
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)


class BrushLayer(lasagne.layers.Layer):

    def __init__(self, incoming, w, h,
                 patch=np.ones((3, 3)), n_steps=10, return_seq=False, **kwargs):
        super(BrushLayer, self).__init__(incoming, **kwargs)
        self.incoming = incoming
        self.w = w
        self.h = h
        self.n_steps = n_steps
        self.patch = patch.astype(np.float32)
        self.return_seq = return_seq

    def get_output_shape_for(self, input_shape):
        if self.return_seq:
            return (input_shape[0],) + (self.n_steps, self.w, self.h)
        else:
            return (input_shape[0],) + (self.w, self.h)

    def get_output_for(self, input, **kwargs):
        def apply_func(X):

            w = self.w
            h = self.h
            pw = self.patch.shape[0]
            ph = self.patch.shape[1]

            gx, gy, sx, sy, log_sigma = (
                X[:, 0], X[:, 1],
                X[:, 2], X[:, 3],
                X[:, 4])

            gx = norm(gx) * w
            gy = norm(gy) * h
            sx = norm(sx) * w
            sy = norm(sy) * h
            sigma = T.exp(log_sigma)

            a, _ = np.indices((w, pw))
            a = a.astype(np.float32)
            a = a.T
            a = theano.shared(a)
            b, _ = np.indices((h, ph))
            b = b.astype(np.float32)
            b = b.T
            b = theano.shared(b)
            # shape of a (pw, w)
            # shape of b (ph, h)
            # shape of sx : (nb_examples,)
            # shape of sy : (nb_examples,)
            ux = gx.dimshuffle(0, 'x') + (T.arange(1, pw + 1) - pw/2. - 0.5) * sx.dimshuffle(0, 'x')
            # shape of ux : (nb_examples, pw)
            a_ = a.dimshuffle('x', 0, 1)
            ux_ = ux.dimshuffle(0, 1, 'x')
            sigma_ = sigma.dimshuffle(0, 'x', 'x')
            Fx = T.exp(-(a_ - ux_) ** 2 / (2 * sigma_ ** 2))#+ 1e-12
            eps = 1e-8
            # that is,  ...(1, pw, w) - (nb_examples, pw, 1) / ... (nb_examples, 1, 1)
            # shape of Fx : (nb_examples, pw, w)

            Fx = Fx / (Fx.sum(axis=2, keepdims=True) + eps)

            #Fx = theano.printing.Print('this is a very important value')(Fx)

            uy = gy.dimshuffle(0, 'x') + (T.arange(1, ph + 1) - ph/2. - 0.5) * sy.dimshuffle(0, 'x')
            # shape of uy : (nb_examples, ph)
            b_ = b.dimshuffle('x', 0, 1)
            uy_ = uy.dimshuffle(0, 1, 'x')
            sigma_ = sigma.dimshuffle(0, 'x', 'x')
            Fy = T.exp(-(b_ - uy_) ** 2 / (2 * sigma_ ** 2)) + 1e-12
            # that is,  ...(1, ph, h) - (nb_examples, ph, 1) / ... (nb_examples, 1, 1)
            # shape of Fy : (nb_examples, ph, h)
            Fy = Fy / (Fy.sum(axis=2, keepdims=True) + eps)
            patch = theano.shared(self.patch)
            # patch : (pw, ph)
            # Fx : (nb_examples, pw, w)
            # Fy : (nb_examples, ph, h)
            o = T.tensordot(patch, Fy, axes=[1, 1])
            # -> shape (pw, nb_examples, h)
            o = o.transpose((1, 2, 0))
            # -> shape (nb_examples, h, pw)
            o = T.batched_dot(o, Fx)
            # -> shape (nb_examples, h, w)
            o = o.transpose((0, 2, 1))
            # -> shape (nb_examples, w, h)
            return o

        def reduce_func(prev, new):
            if self.return_seq:
                return new + prev
            else:
                return prev + new

        output_shape = (input.shape[0],) + (self.w, self.h)
        init_val = T.zeros(output_shape)
        output, _ = recurrent_accumulation(
                # 'time' should be the first dimension
                input.dimshuffle(1, 0, 2),
                apply_func,
                reduce_func,
                init_val,
                self.n_steps)
        output = output.dimshuffle(1, 0, 2, 3)
        if self.return_seq:
            return output
        else:
            return output[:, -1]


def recurrent_accumulation(X, apply_func, reduce_func,
                           init_val, n_steps, **scan_kwargs):

    def step_function(input_cur, output_prev):
        return reduce_func(apply_func(input_cur), output_prev)

    sequences = [X]
    outputs_info = [init_val]
    non_sequences = []

    result, updates = theano.scan(fn=step_function,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=non_sequences,
                                  strict=False,
                                  n_steps=n_steps,
                                  **scan_kwargs)
    return result, updates

class Repeat(lasagne.layers.Layer):
    def __init__(self, incoming, n, **kwargs):
        super(Repeat, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        tensors = [input]*self.n
        stacked = theano.tensor.stack(*tensors)
        dim = [1, 0] + range(2, input.ndim + 1)
        return stacked.dimshuffle(dim)


def over_op(prev, new):
    prev = norm(prev)
    new = norm(new)
    return prev + new * (1 - prev)


if __name__ == '__main__':
    from lasagne import layers
    n_steps = 10
    inp = layers.InputLayer((None, n_steps, 5))
    brush = BrushLayer(inp, 28, 28, n_steps=n_steps)
    X = T.tensor3()
    fn = theano.function([X], layers.get_output(brush, X))

    X_example = np.random.uniform(0, 1, size=(100, n_steps, 5))
    X_example = X_example.astype(np.float32)
    print(fn(X_example).shape)
