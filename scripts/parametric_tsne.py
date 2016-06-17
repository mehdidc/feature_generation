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

def compute_joint_probabilities(samples, batch_size=5000, d=2, perplexity=30, tol=1e-5, verbose=0):    
    # Initialize some variables
    n = samples.shape[0]
    batch_size = min(batch_size, n)
    
    # Precompute joint probabilities for all batches
    if verbose > 0: print('Precomputing P-values...')
    batch_count = int(n / batch_size)
    P = np.zeros((batch_count, batch_size, batch_size))
    for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):   
        curX = samples[start:start+batch_size]                   # select batch
        P[i], beta = x2p(curX, perplexity, tol, verbose=verbose) # compute affinities using fixed perplexity
        P[i][np.isnan(P[i])] = 0                                 # make sure we don't have NaN's
        P[i] = (P[i] + P[i].T) # / 2                             # make symmetric
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


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from lasagne.layers import DenseLayer as Dense
    from lasagne.layers import InputLayer as Input
    from lasagne.layers.helper import get_output, get_all_params
    from lasagne.nonlinearities import rectify, linear
    from lasagne import updates
    import theano
    import matplotlib.pyplot as plt

    floatX = theano.config.floatX

    def iterate_minibatches(inputs, targets=None, batch_size=128, shuffle=False):
        if targets:
            assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            if targets:
                yield inputs[excerpt], targets[excerpt]
            else:
                yield inputs[excerpt]
    data = load_digits()
    batch_size = 256
    X = data['data']
    y = data['target']
    P = compute_joint_probabilities(X, batch_size=batch_size, d=2, perplexity=30, tol=1e-5, verbose=0)

    x = Input((None, X.shape[1]))
    z = Dense(x, num_units=256, nonlinearity=rectify)
    z = Dense(z, num_units=2, nonlinearity=linear)
    z_pred = get_output(z)
    P_real = T.matrix()
    loss = tsne_loss(P_real, z_pred)

    params = get_all_params(z, trainable=True)
    lr = theano.shared(np.array(0.01, dtype=floatX))
    updates = updates.adam(
        loss, params, learning_rate=lr
    )
    train_fn = theano.function([x.input_var, P_real], loss, updates=updates)
    encode = theano.function([x.input_var], z_pred)

    X_train = X
    Y_train = P
    for epoch in range(1000):
        total_loss = 0
        nb = 0
        for xt in iterate_minibatches(X_train, batch_size=batch_size, shuffle=False):
            yt = Y_train[nb]
            total_loss += train_fn(xt, yt)
            nb += 1
        total_loss /= nb
        print('Loss : {}'.format(total_loss))
        if epoch % 100 == 0:
            lr.set_value(np.array(lr.get_value() * 0.5, dtype=floatX))

    z_train = encode(X_train)
    plt.scatter(z_train[:, 0], z_train[:, 1], c=y)
    plt.show()