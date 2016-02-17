import theano.tensor as T


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


def wta_lifetime(percent):

    def apply_(X):
        X_max = X.max(axis=(2, 3), keepdims=True)  # (B, F, 1, 1)
        idx = (1 - percent) * X.shape[0] - 1
        mask = T.argsort(X_max, axis=0) >= idx  # (B, F, 1, 1)
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
