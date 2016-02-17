from lasagnekit.misc.plot_weights import grid_plot
import matplotlib.pyplot as plt
import theano.tensor as T
import theano
from lasagne import layers as L
import numpy as np


def show_filters(capsule, data, layers, w, h, c, **kw):

    W = layers["unconv"].W.get_value().copy()
    W = W.reshape((W.shape[0] * W.shape[1],
                   W.shape[2], W.shape[3]))
    opt = dict(cmap='gray', interpolation='none')
    grid_plot(W, imshow_options=opt)
    plt.show()


def reduc(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):
    from sklearn.manifold import TSNE
    from lasagnekit.datasets.mnist import MNIST
    import matplotlib
    data = MNIST()
    data.load()
    x = T.tensor4()
    name = layer_name
    y = L.get_output(layers[name], x)
    f = theano.function([x], y)

    X = data.X[0:2000]
    y = data.y[0:2000]
    X = X.reshape((X.shape[0], c, w, h))
    z = f(X)
    z = z.max(axis=(2, 3))
    z = z.reshape((z.shape[0], -1))
    model = TSNE(n_components=2)
    # model = PCA(n_components=2)
    l = model.fit_transform(z)

    classes = set(y)
    cols = matplotlib.colors.cnames.keys()
    for i, cl in enumerate(classes):
        selector = (y == cl)
        plt.scatter(l[selector, 0], l[selector, 1], c=cols[i],label=str(cl))
    plt.legend()
    plt.show()


def generative(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):
    name = "wta_channel"
    xgen = T.tensor4()
    x = T.tensor4()
    ygen = L.get_output(layers[name], xgen)
    y = L.get_output(layers[name], x)

    def euc_presv(a, b):
        return ((a - b) ** 2).sum()

    loss = euc_presv(ygen, y)

    grads = theano.function([x, xgen], T.grad(loss, xgen))
    get_loss = theano.function([x, xgen], loss)

    X = data.X
    X = X.reshape((X.shape[0], c, w, h))
    X = X[0:1]

    X_gen = np.random.uniform(size=X.shape)
    X_gen = X_gen.astype(np.float32)
    for i in range(10):
        g = grads(X, X_gen)
        X_gen -= 0.01 * g
        print(get_loss(X, X_gen))

    plt.imshow(X_gen[0:1])
    plt.show()


def neuronviz(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):
    from lasagnekit.misc.plot_weights import tile_raster_images
    name = layer_name

    xgen = T.tensor4()
    y = L.get_output(layers[name], xgen)

    nb_features = layers[name].output_shape[1]

    ex = T.arange(nb_features)
    loss = ((y[ex, ex]) ** 2).sum()

    grads = theano.function([xgen], T.grad(loss, xgen))
    get_loss = theano.function([xgen], loss)

    shape = (nb_features,) + layers["input"].output_shape[1:]

    X_gen = np.random.uniform(size=shape)
    X_gen = X_gen.astype(np.float32)
    for i in range(10000):
        g = np.array(grads(X_gen))
        g /= (np.sqrt(np.mean(np.square(g), axis=(1, 2, 3), keepdims=True)) + 1e-5)
        X_gen += g
        print(get_loss(X_gen))

        if i % 100 == 0:
            img = tile_raster_images(X_gen, data.img_dim[0:2], (10, 10))
            plt.imshow(img, cmap="gray")
            plt.savefig("out.png")
            plt.show()


def genetic(capsule, data, layers, w, h, c,
            layer_name="wta_spatial", nb_iter=100,
            just_get_function=False,
            out="out.png",
            params=None,
            **kw):
    x = T.tensor4()
    if layer_name in layers:
        name = layer_name
    elif "wta_channel" in layers:
        name = "wta_channel"
    elif "wta_spatial" in layers:
        name = "wta_spatial"
    else:
        name = "wta"

    x = T.tensor4()
    g = theano.function(
        [x],
        L.get_output(layers[name], x)
    )
    f = theano.function(
        [x],
        L.get_output(layers["output"], {layers[name]: x}))

    # A is a set of examples (population), the operation
    # should transform it into a new population (not necessarily the same
    # number)
    def identity(A, **kwargs):
        return A

    def random(A, nb=100, k=2):
        shape = (nb,) + A.shape[1:]
        HID = np.zeros(shape).astype(np.float32)
        for a in range(k):
            H = np.random.uniform(size=shape)
            H *= (H == H.max(axis=(1, 2, 3), keepdims=True)) * 100
            H = H.astype(np.float32)
            HID += H
        return HID

    def mutation(A, nb=100, a=0.8, b=0.9999):
        shape = (nb,) + A.shape[1:]
        mask = np.random.uniform(size=shape) <= a
        mask2 = np.random.uniform(size=shape) > b
        s = np.random.randint(0, A.shape[0], size=nb)
        A = A[s]
        A = A * mask + mask2 * 100
        A = A.astype(np.float32)
        return A

    def smarter_mutation(A, born_perc=0.1, dead_perc=0.1, nbtimes=1, nb=100):
        # val = A.max()
        perc = born_perc + dead_perc
        nb_filters = A.shape[1]
        size = int(perc * nb_filters)
        for i in range(nbtimes):
            for a in A:
                indices = np.random.choice(np.arange(len(A)),
                                           size=size, replace=True)
                nb_born = int(born_perc*nb_filters)
                born_indices = indices[0:nb_born]
                dead_indices = indices[nb_born:]
                a[dead_indices] = 0
                for idx in born_indices:
                    a[idx] = 0
                    x, y = np.random.randint(a.shape[1]), np.random.randint(a.shape[2])
                    # val = np.random.choice((0.01, 0.1, 1, 10))
                    #val = np.random.uniform()
                    val = 10
                    a[idx, x, y] = val
        return A

    def switcher(A, nbtimes=1, size=2, nb=100):
        for _ in range(nbtimes):
            for a in A:
                indices = np.random.choice(len(a), size=size, replace=True)
                indices_shuffled = indices.copy()
                np.random.shuffle(indices_shuffled)
                for i, j in zip(indices, indices_shuffled):
                    xi, yi = np.unravel_index(a[i].argmax(), a[i].shape)
                    xj, yj = np.unravel_index(a[j].argmax(), a[j].shape)
                    a[i, xi, yi], a[i, xj, yj] = a[i, xj, yj], a[i, xi, yi]
                    a[j, xi, yi], a[j, xj, yj] = a[j, xj, yj], a[j, xi, yi]
        return A

    def crossover(A, nb=100):
        S = np.zeros((nb,) + A.shape[1:])
        for i in range(nb):
            ind1, ind2 = np.random.randint(0, A.shape[0], size=2)
            s1, s2 = A[ind1], A[ind2]
            s = np.zeros_like(s1)
            c = np.random.randint(0, 2, size=A.shape[1])
            s[c == 0] = s1[c == 0]
            s[c == 1] = s2[c == 1]
            S[i] = s
        return S.astype(np.float32)

    def diversity(X):
        from sklearn.neighbors import NearestNeighbors
        K = params.get("nearest_neighbors", 10)
        X_ = X.reshape((X.shape[0], -1))
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X_)
        distances, _ = nbrs.kneighbors(X_)
        nb = params.get("nb", 100)
        if distances.shape[1] > nb:
            D = distances[:, nb:]
        else:
            D = distances
        return D.mean(axis=1)

    def reconstruction(X):
        return ((X - capsule.reconstruct(X)) ** 2).sum(axis=(1, 2, 3))

    def reconstruction_and_diversity(X):
        return reconstruction(X) + 0.1 * diversity(X)

    def perform():
        # X = data.X
        # X = X.reshape((X.shape[0], c, w, h))
        X = np.random.uniform(size=(data.X.shape[0], c, w, h))
        X = X.astype(np.float32)
        nb = params.get("nb", 100)
        survive = params.get("nb", 20)
        compute_fitness = reconstruction
        # compute_fitness = reconstruction_and_diversity
        feat = g(X)
        evals = compute_fitness(X)
        indices = np.argsort(evals)
        X, evals, feat = X[indices], evals[indices], feat[indices]
        print(evals)

        for i in range(nb_iter):
            #N = 4
            #dvrsty = -diversity(feat[0:survive*N].max(axis=(2, 3)))
            #indices = np.argsort(dvrsty)
            #indices = indices[0:survive]

            indices = np.arange(0, survive) 
            best = X[indices]
            best_evals = evals[indices]
            best_feat = g(best)

            children_feat = crossover(best_feat, nb=nb)
            children_feat = switcher(children_feat)
            children_feat = smarter_mutation(
                    best_feat,
                    born_perc=0.1,
                    dead_perc=0.1,
                    nbtimes=1
            )
            children_px = f(children_feat)
            children_evals = compute_fitness(children_px)

            evals = np.concatenate((best_evals, children_evals), axis=0)
            X = np.concatenate((best, children_px), axis=0)
            feat = np.concatenate((best_feat, children_feat), axis=0)
            indices = np.argsort(evals)
            X, evals, feat = X[indices], evals[indices], feat[indices]
            print("Population mean Fitness : {}".format(evals.mean()))

        # indices = np.argsort(-diversity(feat[0:len(feat)/2].max(axis=(2, 3))))
        # return X[indices][0:nb]
        print(evals[0])
        return X[0:1]

    if just_get_function:
        return perform
    else:
        from lasagnekit.misc.plot_weights import grid_plot
        y = perform()
        fig = plt.figure()
        grid_plot(y[:, 0],
                  imshow_options={"cmap": "gray"},
                  fig=fig)
        plt.savefig(out)


def recons(capsule, data, layers, w, h, c,
           layer_name="wta_spatial",
           **kw):
 
    from lasagnekit.misc.plot_weights import tile_raster_images, grid_plot
    name = layer_name
    x = T.tensor4()
    g = theano.function(
        [x],
        L.get_output(layers[name], x)
    )
    loss = ((L.get_output(layers["output"], x) - x) ** 2).sum(axis=(1, 2, 3)).mean()
    get_loss = theano.function([x], loss)
    get_grad = theano.function([x], theano.grad(loss, x))
    nb_examples = 100
    shape = (nb_examples,) + layers["input"].output_shape[1:]
    x = np.random.uniform(size=shape).astype(np.float32)
    alpha = 0.5
    for i in range(500):
        g = get_grad(x)
        x -= alpha * g
        print(get_loss(x))
    grid_plot(x[:, 0], imshow_options={"cmap": "gray"})
    plt.savefig("out.png")
    plt.show()


def recons_from_features(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):
    from lasagnekit.misc.plot_weights import tile_raster_images, grid_plot
    from scipy.optimize import fmin_l_bfgs_b
    name = layer_name
    x = T.tensor4()
    recons = L.get_output(layers["output"], {layers[name]: x})
    get_recons = theano.function(
        [x],
        recons
    )
    loss = ((recons - L.get_output(layers["output"], recons)) ** 2).sum(axis=(1, 2, 3)).mean()
    get_loss = theano.function(
            [x],
            loss
    )

    get_loss = theano.function([x], loss)
    get_grad = theano.function([x], theano.grad(loss, x))

    nb_examples = 100
    shape = (nb_examples,) + layers[name].output_shape[1:]

    def eval_loss(x0):
        x0_ = x0.reshape(shape).astype(np.float32)
        l = get_loss(x0_)
        return l

    def eval_grad(x0):
        x0_ = x0.reshape(shape).astype(np.float32)
        g = np.array(get_grad(x0_)).flatten().astype(np.float64)
        return g

    x = np.random.uniform(size=shape).astype(np.float32)
    for i in range(500):
        #g = get_grad(x)
        #x -= alpha * np.array(g)
        fmin_l_bfgs_b(eval_loss, x.flatten(), fprime=eval_grad, maxfun=40)
        print(get_loss(x))
        if i % 10 == 0:
            y = get_recons(x)
            grid_plot(y[:, 0], imshow_options={"cmap": "gray"})
            plt.savefig("out.png")
            plt.show()


def activate(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):

    if "wta_channel" in layers:
        name = "wta_channel"
    elif "wta_spatial" in layers:
        name = "wta_spatial"
    else:
        name = "wta"
    x = T.tensor4()
    g = theano.function(
        [x],
        L.get_output(layers[name], x)
    )
    f = theano.function(
        [x],
        L.get_output(layers["output"], {layers[name]: x}))
    shape = layers[name].output_shape[1:]

    N = 100
    shape = (N,) + shape

    X = data.X[data.y == 8]
    HID = np.zeros(shape).astype(np.float32)
    for a in range(2):
        H = np.random.uniform(size=shape)
        H *= (H == H.max(axis=(1, 2, 3), keepdims=True)) * 100
        H = H.astype(np.float32)
        HID += H
    y = f(HID)
    fig = plt.figure()

    grid_plot(y[:, 0],
              imshow_options={"cmap": "gray"},
              fig=fig)
    plt.show()


def prune(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):
    from lasagnekit.datasets.mnist import MNIST
    from lasagnekit.datasets.rescaled import Rescaled
    from lasagnekit.datasets.subsampled import SubSampled
    from lasagnekit.datasets.helpers import load_once
    nb = 1000
    np.random.seed(1)
    data = SubSampled(load_once(MNIST)(), nb)
    data.load()
    X = data.X[data.y==7]
    X = X.reshape((X.shape[0], c, w, h))
    name = "wta_channel"
    x = T.tensor4()
    g = theano.function(
        [x],
        L.get_output(layers[name], x)
    )
    f = theano.function(
        [x],
        L.get_output(layers["output"], {layers[name]: x}))
    A = g(X)
    A = A[0:1]
    print((A>0).sum(axis=1))
    plt.imshow( (A>0).sum(axis=1)[0], cmap="gray", interpolation='none')
    y = f(A)
    a = np.arange(A.shape[1])[A[0].max(axis=(1, 2)) > 0]
    k = 1
    fig = plt.figure(figsize=(10, 10))
    nb = len(a)
    l = int(np.sqrt(nb))
    c = nb / l
    if (nb % l) > 0:
        c += 1
    B = A.copy()
    IM = []
    for i in range(len(a)):
        B[0, a[i]] = 0
        y_ = f(B)
        IM.append(y_[0, 0].tolist())
        plt.subplot(l, c, k)
        plt.axis('off')
        plt.imshow(y_[0, 0], cmap="gray", interpolation='none')
        k += 1
    plt.show()

    IM = np.array(IM)
    from lasagnekit.misc.anim import ImageAnimation
    print(IM.shape)
    anim = ImageAnimation(IM[::-1], interval=100, cmap='gray')
    anim.save("out.mp4")


def interp(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):
    from gui import launch
    from lasagnekit.datasets.mnist import MNIST
    from lasagnekit.datasets.subsampled import SubSampled
    from lasagnekit.datasets.helpers import load_once
    from lasagnekit.misc.plot_weights import tile_raster_images
    nb = 1000
    np.random.seed(12232)
    data = SubSampled(load_once(MNIST)(), nb)
    data.load()
    # name = "wta_channel"
    # name = "input"
    name = layer_name
    x = T.tensor4()
    g = theano.function(
        [x],
        L.get_output(layers[name], x)
    )
    X = data.X
    X = X.reshape((X.shape[0], c, w, h))

    ind_left = np.random.randint(0, nb, size=10)
    ind_right = np.random.randint(0, nb, size=10)
    left = g(X[ind_left])
    right = g(X[ind_right])
    N = 20
    w_ = np.linspace(0, 1, N)
    w_ = w_[None, :, None, None, None]
    left_ = left[:, None, :, :]
    right_ = right[:, None, :, :]
    H = left_ * w_ + right_ * (1 - w_)
    H = H.reshape((H.shape[0] * H.shape[1],
                   H.shape[2], H.shape[3], H.shape[4]))
    H = H.astype(np.float32)
    f = theano.function(
        [x],
        L.get_output(layers["output"], {layers[name]: x}))
    y = f(H)
    #grid_plot(y[:, 0],
    #          imshow_options={"cmap": "gray"},
    #          nbrows=H.shape[0] / N,
    #          nbcols=N,
    #          fig=fig)
    img = tile_raster_images(y[:, 0], (w, h), (H.shape[0] / N, N))
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.savefig("interp.png")
    plt.show()


def myth(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):
    x = T.tensor4()
    g = theano.function(
        [x],
        L.get_output(layers["output"], x)
    )
    dummy = np.zeros((1, c, w, h)).astype(np.float32)
    capsule.batch_optimizer.lr.set_value(0.01)
    try:
        capsule.fit(X=dummy)
    except KeyboardInterrupt:
        print("keyboard interrupt.")
        pass
    im = g(data.X[0:1])
    plt.imshow(im[0, 0], cmap="gray")
    plt.savefig("myth.png")


def gui(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):
    from gui import launch
    launch(capsule, data, w, h, c, max_feature_maps=10)


def ipython(capsule, data, layers, w, h, c, **kw):
    from IPython import embed
    embed()


def beauty(capsule, data, layers, w, h, c, layer_name="wta_spatial", **kw):

    import seaborn as sns
    import pandas as pd
    name = layer_name
    x = T.tensor4()
    loss = ((L.get_output(layers["output"], x) - x) ** 2).sum(axis=(1, 2, 3)).mean()
    # P = L.get_output(layers[name], x)
    P = layers[name].W
    get_grad = theano.function([x], (theano.grad(loss, P)))

    nb = 1000
    x_random = np.random.uniform(size=(nb, c, w, h)).astype(np.float32)

    X = data.X
    x_dataset = X[np.random.randint(0, X.shape[0], size=nb)]
    x_dataset = x_dataset.reshape(x_random.shape)

    # DIRECTION
    from sklearn.decomposition import PCA
    from lpproj import LocalityPreservingProjection

    plt.clf()
    g_random = get_grad(x_random)
    g_dataset = get_grad(x_dataset)
    g = np.concatenate((g_random, g_dataset), axis=0)
    g = g.reshape((g.shape[0], -1))
    pca = LocalityPreservingProjection(n_components=2)
    g = pca.fit_transform(g)
    plt.scatter(g[0:len(g_random), 0], g[0:len(g_random), 1], c='blue', label="random")
    plt.scatter(g[len(g_random):, 0], g[len(g_random):, 1], c='green', label="dataset")
    plt.legend()
    plt.savefig(name+"_direction.png")
    plt.show()
    # NORM
    plt.clf()
    x = []
    y = []
    for i in range(nb):
        g = ((get_grad(x_random[i:i+1])) ** 2).sum()
        x.append("random")
        y.append(g)
        g = (get_grad(x_dataset[i:i+1])**2).sum()
        x.append("mnist")
        y.append(g)
    x = np.array(x)
    y = np.array(y)
    print(y)
    print(x.shape, y.shape)
    df = pd.DataFrame({"data": x, "grad": y})
    sns.boxplot(x="data", y="grad", data=df)
    plt.savefig(name+"_norm.png")
    plt.show()
