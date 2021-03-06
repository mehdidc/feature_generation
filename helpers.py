import theano.tensor as T
import numpy as np
import theano
import os
import lasagne
from layers import FeedbackGRULayer, TensorDenseLayer
from utils.sparsemax_theano import sparsemax
from collections import defaultdict

def get_stat(name, stats):
    return [stat[name] for stat in stats]

def iterate_minibatches(nb_inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(nb_inputs)
        np.random.shuffle(indices)
    for start_idx in range(0, max(nb_inputs, nb_inputs - batchsize + 1), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield excerpt



def norm(x):
    return (x - x.min()) / (x.max() - x.min() + T.eq(x.max(), x.min()) + 1e-12)


def norm_maxmin(x):
    return norm(x)


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

def max_k_spatial(nb=1):

    def apply_(X):
        shape = X.shape
        X_ = X.reshape((X.shape[0] * X.shape[1], X.shape[2] * X.shape[3]))
        idx = T.argsort(X_, axis=1)[:, X_.shape[1] - nb]
        val = X_[T.arange(X_.shape[0]), idx]
        mask = X_ >= val.dimshuffle(0, 'x')
        eps = 1e-10
        X_ = (X_ * mask) / (X_ + eps)
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
        idx = ((1 - percent) * X.shape[0] - 1)############# SHIT!
        mask = T.argsort(X, axis=1) >= idx  # (B, F)
        return X * mask
    return apply_

def wta_fc_sparse_correct(percent):
    def apply_(X):
        idx = ((1 - percent) * X.shape[1] - 1)
        mask = T.argsort(X, axis=1) >= idx  # (B, F)
        return X * mask
    return apply_

def wta_fc_sparse_nb_active(nb_active):
    def apply_(X):
        idx = X.shape[1] - nb_active
        theta = X[T.arange(X.shape[0]), T.argsort(X, axis=1)[:, idx]]
        mask = X >= theta[:, None]
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


def salt_and_pepper(x, rng=np.random, backend='theano', corruption_level=0.5):
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




class BrushLayer(lasagne.layers.Layer):

    def __init__(self, incoming, w, h,
                 patch=np.ones((3, 3)), n_steps=10,
                 return_seq=False,
                 stride=True,
                 sigma=None,
                 reduce_func=lambda x,y:x+y,
                 normalize_func=norm,
                 nonlin_func=lambda x:x,
                 **kwargs):
        super(BrushLayer, self).__init__(incoming, **kwargs)
        self.incoming = incoming
        self.w = w
        self.h = h
        self.n_steps = n_steps
        self.patch = patch.astype(np.float32)
        self.return_seq = return_seq
        self.stride = stride
        self.sigma = sigma
        self.reduce_func_ = reduce_func
        self.norm = normalize_func
        self.nonlin_func = nonlin_func

    def get_output_shape_for(self, input_shape):
        if self.return_seq:
            return (input_shape[0],) + (self.n_steps, self.w, self.h)
        else:
            return (input_shape[0],) + (self.w, self.h)

    def apply_func(self, X):
        w = self.w
        h = self.h
        pw = self.patch.shape[0]
        ph = self.patch.shape[1]

        gx, gy = X[:, 0], X[:, 1],
        # gx : x position
        # gy : y position
        # sx : x stride
        # sy : y stride
        gx = self.norm(gx) * w
        gy = self.norm(gy) * h

        if self.stride is True:
            sx, sy = X[:, 2], X[:, 3]
            sx = self.norm(sx) * w
            sy = self.norm(sy) * h
        else:
            sx = T.ones_like(gx)
            sy = T.ones_like(gx)

        if self.sigma is None:
            log_sigma = X[:, 4]
            sigma = T.exp(log_sigma)
        else:
            sigma = T.ones_like(gx)

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

    def reduce_func(self, prev, new):
        return self.nonlin_func(self.reduce_func_(prev, new))

    def get_output_for(self, input, **kwargs):
        output_shape = (input.shape[0],) + (self.w, self.h)
        init_val = T.zeros(output_shape)
        output, _ = recurrent_accumulation(
            # 'time' should be the first dimension
            input.dimshuffle(1, 0, 2),
            self.apply_func,
            self.reduce_func,
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
    prev = (prev)
    new = (new)
    return prev + new * (1 - prev)

def normalized_sum_op(new, prev):
    eps = 1e-10
    # if new and prev used different strides, this is important.
    new = new - new.min(axis=(1, 2, 2), keepdims=True) 
    new = new / (new.max(axis=(1, 2, 3), keepdims=True) + eps)
    prev = prev - prev.min(axis=(1, 2, 3), keepdims=True) 
    prev = prev / (prev.max(axis=(1, 2, 3), keepdims=True) + eps)
    return new + prev

def sub_op(new, prev):
    return prev - new

def mask_op(new, prev):
    img1 = prev
    img2 = new
    eps=0.1
    a1 = 1 - (T.abs_(img1[:,0])<eps) * (T.abs_(img1[:,1])<eps) * (T.abs_(img1[:,2])<eps)
    a2 = 1 - (T.abs_(img2[:,0])<eps) * (T.abs_(img2[:,1])<eps) * (T.abs_(img2[:,2])<eps)
    a1=a1[:, None, :, :]
    a2=a2[:,None, :, :]
    img = img1 * (1 - a2) + img2 * a2
    return img

def mask_smooth_op(new, prev):
    img1 = prev
    img2 = new
    a1 = img1.mean(axis=1,keepdims=True) 
    a1 = T.cast(a1, 'float32')
    a2 = img2.mean(axis=1,keepdims=True) 
    a2 = T.cast(a2, 'float32')
    #a1=a1[:, None, :, :]
    #a2=a2[:, None, :, :]
    img = img1 * (1 - a2) + img2 * a2
    return img

def normalized_over_op(prev, new):
    prev = (prev)
    new = (new)
    return (prev>0.5) * (new > 0.5) * new + (prev<=0.5)*(new>0.5) * new +  (prev>0.5)*(new<=0.5)*prev+(prev<=0.5)*(new<=0.5)*T.maximum(prev,new)


def correct_over_op(alpha):
    def fn(prev, new):
        return (prev * (1 - alpha) + new) / (2 - alpha)
    return fn


def max_op(prev, new):
    return T.maximum(prev, new)


def thresh_op(theta):
    def fn(prev, new):
        return (new > theta) * new +  (new <= theta ) * prev
    return fn


def sum_op(prev, new):
    # fix this
    return prev + new


def axis_softmax(x, axis=1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out


class GenericBrushLayer(lasagne.layers.Layer):

    def __init__(self, incoming, w, h,
                 patches=np.ones((1, 1, 3, 3)),
                 col='grayscale',
                 n_steps=10,
                 return_seq=False,
                 reduce_func=sum_op,
                 to_proba_func=T.nnet.softmax,
                 normalize_func=T.nnet.sigmoid,
                 x_sigma='predicted',
                 y_sigma='predicted',
                 x_stride='predicted',
                 y_stride='predicted',
                 patch_index='predicted',
                 color='predicted',
                 x_min=0,
                 x_max='width',
                 y_min=0,
                 y_max='height',
                 w_left_pad=0,
                 w_right_pad=0,
                 h_left_pad=0,
                 h_right_pad=0,
                 color_min=0,
                 color_max=1,
                 stride_normalize=False,
                 eps=0,
                 learn_patches=False,
                 coords='continuous',
                 **kwargs):
        """
        w : width of resulting image
        h : height of resulting image
        patches : (nb_patches, color, ph, pw)
        col : 'grayscale'/'rgb' or give the nb of channels as an int
        n_steps : int, nb of time steps
        return_seq : if True returns the seq (nb_examples, n_steps, c, h, w)
                     if False returns (nb_examples, -1, c, h, w)
        reduce_func : function used to update the output, takes prev
                      output as first argument and new output
                      as second one.
        normalize_func : if a function, it is used to normalize between 0 and 1 for :
                            - coordinates
                            - stride if stride=='predicted' (for x and y)
                            - sigma if sigma=='predicted' (for x and y)
                            - color if it is ndarray
                            - color if color=='predicted'. 
                         if a dict, then specify functions separately:
                            {'coords': ..., 'stride': ..., 'sigma': ..., 'color': }
        x_sigma : if 'predicted' taken from input else use the provided
                  value
        y_sigma : if 'predicted' taken from input else use the provided
                  value
        x_stride : if 'predicted' taken from input else use the provided
                   value
        y_stride : if 'predicted' taken from input else use the provided
                   value
        patch_index: if 'predicted' taken from input then apply to_proba_func to
                     obtain probabilities, otherwise it is an int
                     which denotes the index of the chosen patch, that is,
                     patches[patch_index]
        color : if 'predicted' taken from input then merge to patch colors.
                if 'patch' then use patch colors only.
                if ndarray, then we have a discrete number of learned colors,
                and the array represents the initial colors and its shape
                is (nb_colors, nb_col_channels).
                
                otherwise it should be a number if col is 'grayscale' or
                a 3-tuple if col is 'rgb' and then then same color is
                merged to the patches at all time steps. 

        x_min : the minimum value for the coords in the w scale
        x_max : if 'width' it is equal to w, else use the provided value

        y_min : the minimum value for the coords in the w scale
        y_max : if 'height' it is equal to h, else use the provided value
        
        w_left_pad  : int/'half_patch'.
                      augment virtually the resulting image with padding to take into account pixels outside
                      the image to have proper normalization of Fx.
                      if 'half_patch', then the padding is the half of the patch width so that a coordinate
                      of 0, 0 with x_min=0 and x_max='width' will show the bottom right quarter of the patch
        w_right_pad : same than w_left_pad but right of the image
        h_left_pad  : like w_left_pad but for height
        h_right_pad :  like w_right_pad but for height

        color_min : min val of color. this and color_max can be helpful to implement negative colors, negative colors can be used
                    to predicted delta color instead of color so that when canvas are summed up something like opacity could be implemented
                    . for instance if we have two overlapping brushes (first is bigger than second) and we want want with color [1 0 0] and the second
                    [0 1 0], what we can do is to predict the color [1 0 0] for the first and the color [-1 1 0] for the second so that the red
                    component is cancelled.
        color_max : max val of color
        stride_normalize : if True multiply Fx by stride_x and Fy by stride_y, this is useful when summing canvas
                           which has different strides, stride_nornalize makes canvas of different stride on the same scale.
        """
        super(GenericBrushLayer, self).__init__(incoming, **kwargs)
        self.incoming = incoming
        self.w = w
        self.h = h
        self.nb_col_channels = (3 if col == 'rgb' else
                                1 if col == 'grayscale'
                                else col)
        assert self.nb_col_channels in (1, 3)
        self.n_steps = n_steps
        self.patches = patches
        self.return_seq = return_seq

        self.reduce_func = reduce_func
        if not isinstance(normalize_func, dict):
            self.normalize_func = defaultdict(lambda:normalize_func)
        else:
            self.normalize_func = normalize_func
        self.to_proba_func = to_proba_func
        self.x_sigma = x_sigma
        self.y_sigma = y_sigma
        self.x_stride = x_stride
        self.y_stride = y_stride
        self.x_min = x_min
        self.x_max = w if x_max == 'width' else x_max
        self.y_min = y_min
        self.y_max = h if y_max == 'height' else y_max
        self.patch_index = patch_index
        self.color = color
        self.learn_patches = learn_patches
        self.w_left_pad = w_left_pad
        self.w_right_pad = w_right_pad
        self.h_left_pad = h_left_pad
        self.h_right_pad = h_right_pad
        self.color_min = color_min
        self.color_max = color_max
        self.stride_normalize = stride_normalize
        self.coords = coords

        if learn_patches:
            if isinstance(self.patches, np.ndarray):
                shape = self.patches.shape
            else:
                shape = self.patches.get_value().shape
            assert shape[1] == self.nb_col_channels
            self.ph, self.pw = shape[2:]
            self.patches_ = self.add_param(self.patches, shape, name="patches")
        else:
            if isinstance(self.patches, np.ndarray):
                shape = self.patches.shape
            else:
                shape = self.patches.get_value().shape
            assert shape[1] == self.nb_col_channels
            self.ph, self.pw = shape[2:]
            self.patches_ = theano.shared(self.patches)

        if isinstance(self.color, np.ndarray):
            assert self.color.shape[1] == self.nb_col_channels
            self.colors_ = self.add_param(self.color, self.color.shape, name="colors")
        elif isinstance(self.color, theano.compile.SharedVariable):
            assert self.color.get_value().shape[1] == self.nb_col_channels
            self.colors_ = self.add_param(self.color, self.color.get_value().shape, name="colors")
        self.eps = eps
        self._nb_input_features = incoming.output_shape[2]
        self.assign_ = {}

    def get_output_shape_for(self, input_shape):
        if self.return_seq:
            return (input_shape[0], self.n_steps,
                    self.nb_col_channels, self.w, self.h)
        else:
            return (input_shape[0], self.nb_col_channels, self.w, self.h)

    def apply_func(self, X):
        w = self.w
        h = self.h
        nb_patches = self.patches.shape[0]
        ph = self.ph
        pw = self.pw
        nb_features = self._nb_input_features
        pointer = 0
        if self.coords == 'continuous':
            gx, gy = X[:, 0], X[:, 1]
            gx = self.normalize_func['coords'](gx) * (self.x_max - self.x_min) + self.x_min
            gy = self.normalize_func['coords'](gy) * (self.y_max - self.y_min) + self.y_min
            self.assign_['gx'] = 0
            self.assign_['gy'] = 1
            pointer += 2
        elif self.coords == 'discrete':
            nx = self.x_max - self.x_min
            cx = theano.shared(np.linspace(0, 1, nx).astype(np.float32))
            gx_pr = X[:, pointer:pointer + nx]
            gx_pr = self.to_proba_func(gx_pr)
            gx = T.dot(gx_pr, cx)
            gx = gx * (self.x_max - self.x_min) + self.x_min
            self.assign_['gx'] = (pointer, pointer + nx)
            pointer += nx
            ny = self.y_max - self.y_min
            cy = theano.shared(np.linspace(0, 1, ny).astype(np.float32))
            gy_pr = X[:, pointer:pointer + ny]
            gy_pr = self.to_proba_func(gy_pr)
            gy = T.dot(gy_pr, cy)
            gy = gy * (self.y_max - self.y_min) + self.y_min
            self.assign_['gy'] = (pointer, pointer + ny)
            pointer += ny
        else:
            raise Exception('invalid value : {} for coords'.format(self.coords))
        if self.x_stride == 'predicted':
            sx = X[:, pointer]
            sx = self.normalize_func['stride'](sx)
            self.assign_['x_stride'] = pointer
            pointer += 1
        elif type(self.x_stride) == list:
            xs = (np.array(self.x_stride).astype(np.float32))
            xs_pr = X[:, pointer:pointer + len(xs)]
            xs_pr = self.to_proba_func(xs_pr)
            sx = T.dot(xs_pr, xs)
            self.assign_['x_stride'] = (pointer, pointer + len(xs))
            pointer += len(xs)
        else:
            sx = T.ones_like(gx) * self.x_stride

        if self.y_stride == 'predicted':
            sy = X[:, pointer]
            sy = self.normalize_func['stride'](sy)
            self.assign_['y_stride'] = pointer
            pointer += 1
        elif type(self.y_stride) == list:
            ys = (np.array(self.y_stride).astype(np.float32))
            ys_pr = X[:, pointer:pointer + len(ys)]
            ys_pr = self.to_proba_func(ys_pr)
            sy = T.dot(ys_pr, ys)
            self.assign_['y_stride'] = (pointer, pointer + len(ys))
            pointer += len(ys)
        else:
            sy = T.ones_like(gy) * self.y_stride

        if self.x_sigma == 'predicted':
            log_x_sigma = X[:, pointer]
            x_sigma = T.exp(log_x_sigma)
            x_sigma = self.normalize_func['sigma'](log_x_sigma) * pw
            self.assign_['x_sigma'] = pointer
            pointer += 1
        elif type(self.x_sigma) == list:
            xs = (np.array(self.x_sigma).astype(np.float32))
            xs_pr = X[:, pointer:pointer + len(xs)]
            xs_pr = self.to_proba_func(xs_pr) * xs
            xs_pr = xs_pr.sum(axis=1)
            x_sigma = xs_pr
            self.assign_['x_sigma'] = (pointer, pointer + len(xs))
            pointer += len(xs)
        else:
            x_sigma = T.ones_like(gx) * self.x_sigma

        if self.y_sigma == 'predicted':
            log_y_sigma = X[:, pointer]
            y_sigma = T.exp(log_y_sigma)
            y_sigma = self.normalize_func['sigma'](log_y_sigma) * ph
            self.assign_['y_sigma'] = pointer
            pointer += 1
        elif type(self.y_sigma) == list:
            ys = (np.array(self.y_sigma).astype(np.float32))
            ys_pr = X[:, pointer:pointer + len(ys)]
            ys_pr = self.to_proba_func(ys_pr) * ys
            ys_pr = ys_pr.sum(axis=1)
            y_sigma = ys_pr
            self.assign_['y_sigma'] = (pointer, pointer + len(ys))
            pointer += len(ys)
        else:
            y_sigma = T.ones_like(gy) * self.y_sigma

        if self.patch_index == 'predicted':
            patch_index = X[:, pointer:pointer + nb_patches]
            self.assign_['patch_index'] = (pointer, pointer + nb_patches)
            pointer += nb_patches
        else:
            patch_index = self.patch_index
        if isinstance(self.color, np.ndarray) or isinstance(self.color, theano.compile.SharedVariable):
            if isinstance(self.color, theano.compile.SharedVariable):
                shape = self.color.get_value().shape
            else:
                shape = self.color.shape
            nb = shape[0]
            colors_pr = X[:, pointer:pointer + nb]#(nb_examples, nb_colors)
            colors_pr = self.to_proba_func(colors_pr) # (nb_examples, nb_colors)
            colors_mix = colors_pr.dimshuffle(0, 1, 'x') * self.colors_.dimshuffle('x', 0, 1) #(nb_examples, nb_colors, 1) * (1, nb_colors, nb_col_channels) = (nb_examples, nb_colors, nb_col_channels)
            colors = colors_mix.sum(axis=1) #(nb_examples, nb_col_channels)
            colors = self.normalize_func['color'](colors) * (self.color_max - self.color_min) + self.color_min
            self.assign_['color'] = (pointer, pointer + nb)
            pointer += nb
        elif self.color == 'predicted':
            colors = X[:, pointer:pointer + self.nb_col_channels]
            colors = self.normalize_func['color'](colors) * (self.color_max - self.color_min) + self.color_min
            self.assign_['color'] = (pointer, pointer + self.nb_col_channels)
            pointer += self.nb_col_channels
        elif self.color == 'patches':
            colors = T.ones((1, 1, 1, 1))
        else:
            assert len(self.color) == self.nb_col_channels
            colors = self.color

        assert nb_features >= pointer, "The number of input features to Brush should be {} instead of {} (or at least bigger)".format(pointer, nb_features)
        
        if self.w_left_pad and self.w_right_pad:
            w_left_pad = self.w_left_pad
            if w_left_pad == 'half_patch':
                w_left_pad = pw / 2
            w_right_pad = self.w_right_pad
            if w_right_pad == 'half_patch':
                w_right_pad = pw / 2
            a, _ = np.indices((w + w_left_pad + w_right_pad, pw)) - w_left_pad
        else:
            w_left_pad = 0
            w_right_pad = 0
            a, _ = np.indices((w, pw))

        a = a.astype(np.float32)
        a = a.T
        a = theano.shared(a)

        if self.w_left_pad and self.w_right_pad:
            h_left_pad = self.h_left_pad
            if h_left_pad == 'half_patch':
                h_left_pad = ph / 2
            h_right_pad = self.h_right_pad
            if h_right_pad == 'half_patch':
                h_right_pad = ph / 2
            b, _ = np.indices((h + h_left_pad + h_right_pad, pw)) - h_left_pad
        else:
            h_left_pad = 0
            h_right_pad = 0
            b, _ = np.indices((h, ph))
        b = b.astype(np.float32)
        b = b.T
        b = theano.shared(b)
        # shape of a (pw, w)
        # shape of b (ph, h)
        # shape of sx : (nb_examples,)
        # shape of sy : (nb_examples,)
        ux = (gx.dimshuffle(0, 'x') +
              (T.arange(1, pw + 1) - pw/2. - 0.5) * sx.dimshuffle(0, 'x'))
        # shape of ux : (nb_examples, pw)
        a_ = a.dimshuffle('x', 0, 1)
        ux_ = ux.dimshuffle(0, 1, 'x')

        x_sigma_ = x_sigma.dimshuffle(0, 'x', 'x')
        y_sigma_ = y_sigma.dimshuffle(0, 'x', 'x')

        Fx = T.exp(-(a_ - ux_) ** 2 / (2 * x_sigma_ ** 2))
        Fx = Fx / (Fx.sum(axis=2, keepdims=True) + self.eps)
        if self.stride_normalize:
            Fx = Fx * sx.dimshuffle(0, 'x', 'x')
        if w_left_pad and w_right_pad:
            Fx = Fx[:, :, w_left_pad:-w_right_pad]
        uy = (gy.dimshuffle(0, 'x') +
              (T.arange(1, ph + 1) - ph/2. - 0.5) * sy.dimshuffle(0, 'x'))
        # shape of uy : (nb_examples, ph)
        b_ = b.dimshuffle('x', 0, 1)
        uy_ = uy.dimshuffle(0, 1, 'x')
        Fy = T.exp(-(b_ - uy_) ** 2 / (2 * y_sigma_ ** 2))
        # shape of Fy : (nb_examples, ph, h)
        Fy = Fy / (Fy.sum(axis=2, keepdims=True) + self.eps)
        if self.stride_normalize:
            Fy = Fy * sy.dimshuffle(0, 'x', 'x')
        if h_left_pad and h_right_pad:
            Fy = Fy[:, :, h_left_pad:-h_right_pad]
        
        patches = self.patches_
        # patches : (nbp, c, ph, pw)
        # Fy : (nb_examples, ph, h)
        # Fx : (nb_examples, pw, w)
        o = T.tensordot(patches, Fy, axes=[2, 1])
        # -> shape (nbp, c, pw, nb_examples, h)
        o = o.transpose((3, 0, 1, 4, 2))
        # -> shape (nb_examples, nbp, c, h, pw)
        o = T.batched_tensordot(o, Fx, axes=[4, 1])
        # -> shape (nb_examples, nbp, c, h, w)

        if self.patch_index == 'predicted':
            patch_index_ = self.to_proba_func(patch_index)
            patch_index_ = patch_index_.dimshuffle(0, 1, 'x', 'x', 'x')
            o = o * patch_index_
            o = o.sum(axis=1)
            # -> shape (nb_examples, c, h, w)
        else:
            o = o[:, patch_index]
            # -> shape (nb_examples, c, h, w)

        if isinstance(self.color, np.ndarray) or isinstance(self.color, theano.compile.SharedVariable):
            o = o * colors.dimshuffle(0, 1, 'x', 'x')
        elif self.color == 'predicted':
            o = o * colors.dimshuffle(0, 1, 'x', 'x')
        elif self.color == 'patches':
            pass
        else:
            colors_ = theano.shared(np.array(colors).astype(theano.config.floatX))
            colors_ = colors_.dimshuffle('x', 0, 'x', 'x')
            o = o * colors_
        return o

    def reduce_func(self, prev, new):
        return self.reduce_func(prev, new)

    def get_output_for(self, input, **kwargs):
        output_shape = (
            (input.shape[0],) +
            (self.nb_col_channels, self.h, self.w))
        init_val = T.zeros(output_shape)
        init_val = T.unbroadcast(init_val, 0, 1, 2, 3)
        # the above single line is to avoid this error:
        # "an input and an output are associated with the same recurrent state
        # and should have the same type but have type 'CudaNdarrayType(float32,
        # (False, True, False, False))' and 'CudaNdarrayType(float32, 4D)'
        # respectively.'))"
        output, _ = recurrent_accumulation(
            # 'time' should be the first dimension
            input.dimshuffle(1, 0, 2),
            self.apply_func,
            self.reduce_func,
            init_val,
            self.n_steps)
        output = output.dimshuffle(1, 0, 2, 3, 4)
        if self.return_seq:
            return output
        else:
            return output[:, -1]

def one_step_brush_layer(*args, **kwargs):
    return GenericBrushLayer(n_steps=1, return_seq=False, *args, **kwargs)[:, 0, :, :, :]

def test_generic_batch_layer():
    from lasagne import layers
    from tools.plot_weights import dispims_color
    from skimage.io import imsave
    n_steps = 10
    nb_features = 10
    inp = layers.InputLayer((None, n_steps, nb_features))
    p1 = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    p2 = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    patches = np.zeros((2, 3, 5, 5))
    patches[0, :] = np.array(p1)
    patches[1, :] = np.array(p2)
    #patches[:] = 1.
    patches = patches.astype(np.float32)
    w, h = 64, 64
    pad = 10
    brush = GenericBrushLayer(
        inp,
        w, h,
        n_steps=n_steps,
        patches=patches,
        col='rgb',
        return_seq=False,
        reduce_func=over_op,
        to_proba_func=sparsemax,
        normalize_func=T.nnet.sigmoid,
        x_sigma=1,#'predicted',
        y_sigma=1,#'predicted',
        x_stride='predicted',
        y_stride='predicted',
        patch_index='predicted',
        color='predicted',
        #color=(1., 0, 0),
        x_min=0,
        x_max='width',
        y_min=0,
        y_max='height',
        w_left_pad='half_patch',
        w_right_pad='half_patch',
        h_left_pad='half_patch',
        h_right_pad='half_patch',
        eps=0)
    X = T.tensor3()
    fn = theano.function([X], layers.get_output(brush, X))
    X_example = np.random.normal(0, 1, size=(100, n_steps, nb_features))
    X_example = X_example.astype(np.float32)
    y = fn(X_example)
    print(y.min(), y.max())
    y = y.transpose((0, 2, 3, 1))
    img = dispims_color(y, border=1, bordercolor=(0.3, 0.3, 0.3))
    imsave('out.png', img)


class DataGen(object):

    def __init__(self, gen_func, nb_chunks=1, batch_size=128):
        self.cnt = 0
        self.nb_chunks = nb_chunks
        self.batch_size = batch_size
        self.gen_func = gen_func

    def load(self):
        if self.cnt == self.nb_chunks or self.cnt == 0:
            X = self.gen_func(self.batch_size * self.nb_chunks)
            X = X.reshape((X.shape[0], -1))
            self.X_cache = X
            self.cnt = 0
        start = self.cnt * self.batch_size
        print(start, self.batch_size, self.X_cache.shape)
        self.X = self.X_cache[start:start + self.batch_size]
        self.cnt += 1


class GaussianSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(nn.random.get_rng().randint(1, 2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
               self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)


class ExpressionLayerMulti(lasagne.layers.MergeLayer):
    def __init__(self, incomings, function, output_shape=None, **kwargs):
        super(ExpressionLayerMulti, self).__init__(incomings, **kwargs)

        if output_shape is None:
            self._output_shape = None
        elif output_shape == 'auto':
            self._output_shape = 'auto'
        elif hasattr(output_shape, '__call__'):
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.function = function

    def get_output_shape_for(self, input_shapes):
        if self._output_shape is None:
            return input_shapes[0]
        elif self._output_shape is 'auto':
            input_shape = input_shapes[0]
            input_shape = (0 if s is None else s for s in input_shape)
            X = theano.tensor.alloc(0, *input_shapes)
            output_shape = self.function(X).shape.eval()
            output_shape = tuple(s if s else None for s in output_shape)
            return output_shape
        else:
            return self._output_shape

    def get_output_for(self, inputs, **kwargs):
        return self.function(*inputs)



def gaussian_log_likelihood(tgt, mu, ls):
    return (-(np.float32(0.5 * np.log(2 * np.pi)) + ls)
            - 0.5 * T.sqr(tgt - mu) / T.exp(2 * ls))


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

def axify(fn):

    def fn_(x, axis=1):
        x = x.transpose((0, 2, 3 , 4, 1))
        x = T.cast(x, theano.config.floatX)
        shape = x.shape
        x = x.reshape((-1, x.shape[-1]))
        x = fn(x)
        x = x.reshape(shape)
        x = x.transpose((0, 4, 1, 2, 3))
        return x
    return fn_

class DataFutureWrapper(object):

    def __init__(self, iterator):
        self.iterator = iterator

    def load(self):
        vals = next(self.iterator)
        for k, v in vals.items():
            setattr(self, k, v)

def floatX(X):
    return np.array(X, dtype=theano.config.floatX)

def grad_noise(algo, loss_or_grads, params, rng=None, n=1, gamma=0.55, **kw):
    updates = algo(loss_or_grads, params, **kw)
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    t = t_prev + 1
    std = T.sqrt(n / (t ** gamma))
    for p in params:
        updates[p] = updates[p] + rng.normal(size=p.shape, avg=0, std=std)
    updates[t_prev] = t
    return updates

def sample_multinomial(X, rng):
    # X is a matrix of (N, m) where N is the number of examples and m numbers representing probabilities(they should sum to 1 for each example).
    # samples from each multinomial distrib defined for each example, and returns a one hot vector containing the selected feature.
    r = rng.uniform(size=(X.shape[0], 1))
    r = T.extra_ops.repeat(r, X.shape[1], axis=1)
    zero_col = T.zeros((X.shape[0], 1))
    X = T.concatenate((zero_col, X), axis=1)
    X = T.extra_ops.cumsum(X, axis=1)
    x = X[:, 0:-1]
    x_next = X[:, 1:]
    in_interval = (r >= x) * (r < x_next)
    return in_interval

if __name__ == '__main__':
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    #test_generic_batch_layer()
    rng = RandomStreams(1)
    x = T.matrix()
    fn = theano.function([x], sample_multinomial(x, rng))
    e = floatX(np.zeros((10000, 2)))
    e[:, 0] = 0.3
    e[:, 1] = 0.7
    print(fn(e).mean(axis=0))
