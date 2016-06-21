# Code adapted from https://github.com/kylemcdonald/Parametric-t-SNE
import numpy as np
import theano.tensor as T

def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
    P = P / sumP
    return H, P

def x2p(X, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
    # Initialize some variables
    n = X.shape[0]                     # number of instances
    P = np.zeros((n, n))               # empty probability matrix
    beta = np.ones(n)                  # empty precision vector
    logU = np.log(u)                   # log of perplexity (= entropy)

    # Compute pairwise distances
    if verbose > 0: print('Computing pairwise distances...')
    sum_X = np.sum(np.square(X), axis=1)
    # note: translating sum_X' from matlab to numpy means using reshape to add a dimension
    D = sum_X + sum_X[:,None] + -2 * X.dot(X.T)

    # Run over all datapoints
    if verbose > 0: print('Computing P-values...')
    for i in range(n):

        if verbose > 1 and print_iter and i % print_iter == 0:
            print('Computed P-values {} of {} datapoints...'.format(i, n))

        # Set minimum and maximum values for precision
        betamin = float('-inf')
        betamax = float('+inf')

        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, indices]
        H, thisP = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, indices] = thisP

    if verbose > 0:
        print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
        print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
        print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))

    return P, beta

def compute_joint_probabilities(samples, batch_size=5000, d=2, perplexity=30, tol=1e-5, verbose=0, P=None):
    # Initialize some variables
    n = samples.shape[0]
    batch_size = min(batch_size, n)

    # Precompute joint probabilities for all batches
    if verbose > 0: print('Precomputing P-values...')
    batch_count = int(n / batch_size)
    if P is None:
        P = np.zeros((batch_count, batch_size, batch_size))
    for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):
        curX = samples[start:start+batch_size]                   # select batch
        P[i], beta = x2p(curX, perplexity, tol, verbose=verbose) # compute affinities using fixed perplexity
        P[i][np.isnan(P[i])] = 0                                 # make sure we don't have NaN's
        P[i] = (P[i] + P[i].T)                                   # make symmetric
        P[i] = P[i] / P[i].sum()                                 # obtain estimation of joint probabilities
        P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)

    return P


# P is the joint probabilities for this batch (Keras loss functions call this y_true)
# activations is the low-dimensional output (Keras loss functions call this y_pred)
def tsne_loss(P, activations):
    d = activations.shape[1]
    v = d - 1.
    eps = 10e-15 # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)
    sum_act = T.sum(T.square(activations), axis=1)
    Q = sum_act.reshape((-1, 1)) + -2 * T.dot(activations, activations.T)
    Q = (sum_act + Q) / v
    Q = T.pow(1 + Q, -(v + 1) / 2)
    Q *= 1 - T.eye(activations.shape[0])
    Q /= T.sum(Q)
    Q = T.maximum(Q, eps)
    C = T.log((P + eps) / (Q + eps))
    C = T.sum(P * C)
    return C


def save_model(net, filename):
    import cPickle as pickle
    fd = open(filename, 'w')
    pickle.dump(net, fd)
    fd.close()


def load_model(filename):
    import cPickle as pickle
    fd = open(filename)
    net = pickle.load(fd)
    fd.close()
    return net

def iterate_minibatches(nb, batch_size=128, shuffle=False):
    if shuffle:
        indices = np.arange(nb)
        np.random.shuffle(nb)
    for start_idx in range(0, nb - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield excerpt

if __name__ == '__main__':
    from lasagne.layers import DenseLayer as Dense
    from lasagne.layers import InputLayer as Input
    from lasagne.layers.helper import get_output, get_all_params
    from lasagne.nonlinearities import rectify, linear, tanh
    from lasagne import updates
    import theano
    import h5py
    from time import time
    data = h5py.File('dataset/generated_images.hdf5')
    X = data['X']
    nb = X.attrs['nb']
    batch_size = 4096
    floatX = theano.config.floatX

    x = Input((None, X.shape[1]))
    z = Dense(x, num_units=500, nonlinearity=tanh)
    #z = Dense(z, num_units=500, nonlinearity=rectify)
    #z = Dense(z, num_units=2000, nonlinearity=rectify)
    z = Dense(z, num_units=2, nonlinearity=linear)
    net = z

    z_pred = get_output(z)
    P_real = T.matrix()
    loss = tsne_loss(P_real, z_pred)

    params = get_all_params(z, trainable=True)
    lr = theano.shared(np.array(0.00001, dtype=floatX))
    updates = updates.adam(
        loss, params, learning_rate=lr
    )
    train_fn = theano.function([x.input_var, P_real], loss, updates=updates)
    encode = theano.function([x.input_var], z_pred)
    avg_loss = 0.
    nb_updates = 0

    P = np.empty((1, batch_size, batch_size))
    for epoch in range(1000):
        total_loss = 0
        nb = 0
        for mb in iterate_minibatches(len(X),
                                      batch_size=batch_size,
                                      shuffle=False):
            t = time()
            xt = X[mb]
            yt = compute_joint_probabilities(xt, batch_size=batch_size, d=2, perplexity=50, tol=1e-5, verbose=0, P=P)
            yt = yt[0]
            if np.any(np.isnan(yt)):
                print('nan')
                continue
            xt = xt.astype(floatX)
            yt = yt.astype(floatX)
            loss = train_fn(xt, yt)
            avg_loss = 0.999 * avg_loss + (1 - 0.999) * loss
            total_loss += loss
            dt = time() - t
            print('Avg loss : {}, nb updates : {}, time : {}'.format(avg_loss, nb_updates, dt))
            nb += 1
            nb_updates += 1
        total_loss /= nb
        print('Loss : {}'.format(total_loss))
        if epoch % 100 == 0:
            lr.set_value(np.array(lr.get_value() * 0.5, dtype=floatX))
        save_model(net, 'tsne.pkl')
    #z_train = encode(X_train)
    #plt.scatter(z_train[:, 0], z_train[:, 1], c=y)
    #plt.savefig('tsne.png')
