from lasagnekit.misc.plot_weights import grid_plot
import matplotlib.pyplot as plt
import theano.tensor as T
import theano
from lasagne import layers as L
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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

def simple_genetic(capsule, data, layers, w, h, c, **params):
    op_params = params.get("op_params", [{"nb": 100}])
    op_names = params.get("op_names", ["mutation"])
    flatten  = params.get("flatten", False)
    assert len(op_names) == len(op_params)

    ops = {
        "mutation": mutation,
        "crossover": crossover,
        "switcher": switcher,
    }

    def apply_op(f):
        for op_name, op_param in zip(op_names, op_params):
            logger.info("Applying {}...".format(op_name))
            op_func = ops[op_name]
            f = op_func(f, **op_param)
        return f

    layer_name = params.get("layer_name", "conv3")

    out = params.get("out", "out.png")
  
    x = T.tensor4()
  
    logger.info("Compiling functions...")
    x = T.tensor4()
    px_to_code = theano.function(
        [x],
        L.get_output(layers[layer_name], x)
    )
    code_to_px = theano.function(
        [x],
        L.get_output(layers["output"], {layers[layer_name]: x}))

    logger.info("Applying genetic op:")

    
    initial = params.get("initial_source", "dataset")
    initial_size = params.get("initial_size", 1)

    if initial == "dataset":
        px_batch = []
        for i in range(initial_size):
            data.load()
            shape = (data.X.shape[0], c, w, h)
            px_batch.append(data.X.reshape(shape))
        px = np.concatenate(px_batch, axis=0)
    elif initial == "random":
        shape = (data.X.shape[0] * initial_size, c, w, h)
        px = np.random.uniform(size=shape)
        px = px.astype(np.float32)

    
    code = px_to_code(px)

    if flatten:
        shape = code.shape[1:]
        code = code.reshape((code.shape[0], -1))
        new_code = apply_op(code)
        new_code = new_code.reshape((new_code.shape[0],) + shape)
        code = code.reshape((code.shape[0],) + shape)
    else:
        new_code = apply_op(code)

    if layer_name == "input":
        new_px = new_code
    else:
        new_px = code_to_px(new_code)

    if params.get("filter", False):
        from sklearn.svm import OneClassSVM
        m = OneClassSVM(verbose=1, nu=0.9)
        new_code = px_to_code(new_px)
        D = np.concatenate((new_code, code), axis=0)
        D = D.reshape((D.shape[0], -1))
        m.fit(D)
        is_novel = (m.predict(D) == 1)
        is_novel = is_novel[0:len(new_code)]
        logger.info("Ratio of novel : {}".format(is_novel.sum()/float(len(is_novel))))
        new_px = new_px[is_novel]

    logger.info("Saving image...")

    if c == 1:
        from lasagnekit.misc.plot_weights import tile_raster_images
        from skimage.io import imsave
        new_px = new_px[:, 0, :, :]
        nb_samples = new_px.shape[0]
        size = int(np.sqrt(nb_samples))
        img = tile_raster_images(new_px, img_shape=(w, h), tile_shape=(size, size))
        imsave(out, img)
    else:
        pass # next time



def genetic(capsule, data, layers, w, h, c,
            **params):
    print(layers.keys())

    allowed_params = set([
        "layer_name",
        "nb_iter",
        "out",
        "just_get_function",
        "nearest_neighbors",
        "tradeoff",
        "fitness_name",
        "arch",
        "modelfile",
        "category",
        "nb_initial",
        "initial_source",
        "nbchildren",
        "nbsurvive",
        "strategy",
        "temperature",
        "born_perc",
        "dead_perc",
        "dead_perc",
        "nbtimes",
        "mutationval",
        "tsne",
        "tsnecentroids",
        'tsneparams',
        "tsnefile",
        "groupshow",
        "save_all",
        "save_all_folder",
        "seed",
        "n_clusters",
        "image_scatter",
        "image_scatter_out",
        "flatten",
        "recons"
    ])
    for p in params.keys():
        assert p in allowed_params, "'{}' not recognizable parameter".format(p)
  
    layer_name = params.get("layer_name", "wta_spatial")
    name = layer_name
    nb_iter = params.get("nb_iter", 100)
    just_get_function = params.get("just_get_function", False)
    out = params.get("out", "out.png")
  
    x = T.tensor4()
  
    logger.info("Compiling functions...")
    x = T.tensor4()
    g = theano.function(
        [x],
        L.get_output(layers[name], x)
    )
    f = theano.function(
        [x],
        L.get_output(layers["output"], {layers[name]: x}))

   

    def vect(F, only_max=True):
        if only_max:
            F = F.max(axis=(2, 3))
        F = F.reshape((F.shape[0], -1))
        return F
    """
    in all the following fitness functions, the smaller the value
    the better
    """
    def nearestneighbours_distance(X, orig=None, feat=None, orig_feat=None):
        assert feat is not None
        from sklearn.neighbors import NearestNeighbors
        K = params.get("nearest_neighbors", 8)
        X_ = vect(feat)
        if orig_feat is not None:
            O_ = vect(orig_feat)
            S = np.concatenate((X_, O_), axis=0)
        else:
            S = X_
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(S)
        distances, _ = nbrs.kneighbors(S)
        if orig is not None:
            D = distances[0:len(X_), :]
        else:
            D = distances
        return D.mean(axis=1)

    def diversity(X, orig=None, feat=None, orig_feat=None):
        return -nearestneighbours_distance(
                X, orig=orig,
                feat=feat,
                orig_feat=orig_feat)

    rec_model = None
    rec_class = None

    def recognizability(X, feat=None, orig=None, orig_feat=None):
        # maximize prediction of some category
        #X_ = X.reshape((X.shape[0], -1))
        X_ = X
        if rec_class == 'any':
            pred = rec_model.predict_proba(X_).max(axis=1)
            return 1 - pred
        else:
            pred = rec_model.predict_proba(X_)[:, rec_class]
            return 1. - pred

    def reconstruction(X, feat=None, orig=None, orig_feat=None):
        return ((X - capsule.reconstruct(X)) ** 2).sum(axis=(1, 2, 3))

    def reconstruction_and_diversity(X, feat=None, orig=None, orig_feat=None):
        diversity = nearestneighbours_distance(X, feat=feat, orig=orig,
                                               orig_feat=orig_feat)
        # minimize reconstruction and maximize  diversity
        return reconstruction(X, orig=orig) - params.get("tradeoff", 0.01) * diversity

    def pngsize(X, feat=None, orig=None, orig_feat=None):
        from cStringIO import StringIO
        import png
        mode = 'L' if c == 1 else 'RGB'
        sizes = []
        X_ = X.transpose((0, 2, 3, 1))
        for x in X_:
            stream = StringIO()
            png.from_array((x*255.).astype(np.int16), mode=mode).save(stream)
            size = len(stream.getvalue())
            sizes.append(size)
        return np.array(sizes)

    def apply_genetic(best_feat, nb=100):            
        children_feat = crossover(best_feat, nb=nb)
        children_feat = switcher(children_feat)
        children_feat = mutation(
                best_feat,
                born_perc=params.get("born_perc", 0.1),
                dead_perc=params.get("dead_perc", 0.1),
                nbtimes=params.get("nbtimes", 1),
                val=params.get("mutationval", 10)
        )
        return children_feat

    # Choose and init fitness
    logger.info("Init fitness...")
    
    fitness_name = params.get("fitness_name", "reconstruction")
    fitness = {
        "reconstruction": reconstruction,
        "reconstruction_and_diversity": reconstruction_and_diversity,
        "recognizability": recognizability,
        "diversity": diversity,
        "pngsize": pngsize
    }
    compute_fitness = fitness[fitness_name]

    if compute_fitness == recognizability:
        rec_model = capsule
        rec_class = params.get("category", "any")
        #from keras.models import model_from_json
        #logger.info("Load recognizability model...")
        #arch = params.get("arch", "models/mnist.json")
        #modelfile = params.get("modelfile", "models/mnist.hdf5")
        #rec_model = model_from_json(open(arch).read())
        #rec_model.load_weights(modelfile)
        #rec_class = params.get("category", "any")

    def perform():

        # Init data
        nb_initial = params.get("nb_initial", data.X.shape[0])
        initial_source = params.get("initial_source", "random")
    
        # nb_initial = 10       
        if initial_source == "dataset":
            X = data.X.reshape((data.X.shape[0], c, w, h))
            X = X[0:nb_initial]
        elif initial_source == "random":
            X = np.random.uniform(size=(data.X.shape[0], c, w, h))
            X = X[0:nb_initial]
        elif initial_source == "centroids":
            categories = list(set(data.y))
            print(categories)
            centroid = np.zeros((len(categories), c, w, h))
            centroid_size = [0] * len(categories)
            for i in range(10):
                data.load()
                X = data.X.reshape((data.X.shape[0], c, w, h))
                for idx, cat in enumerate(categories):
                    S = X[data.y==cat]
                    centroid[idx] += S.sum(axis=0)
                    centroid_size[idx] += len(S)

            for ctroid, ctroidsz in zip(centroid, centroid_size):
                print(ctroidsz, ctroid.max())
                ctroid /= ctroidsz

            X = np.array(centroid).astype(np.float32)
            nb_initial = X.shape[0]
        else:
            raise Exception("bad initial")

        X = X.astype(np.float32)

        # genetic params
        nb = params.get("nbchildren", 100)  # nb of children per iteration
        survive = params.get("nbsurvive", 20)

        # Init genetic
        logger.info("Compute fitness of initial population...")
        feat = g(X)
        evals = compute_fitness(X, feat=feat)
        indices = np.argsort(evals)
        X, evals, feat = X[indices], evals[indices], feat[indices]
        print(evals[0:10])

        archive_px = X.copy()  # all generated images in pixel space
        archive_feat = feat.copy()  # all generated images in feature space
        archive_evals = evals.copy()  # all evaluations of genereted images
        archive_popul_indices = indices  # current indices on archive of the population
        centroids = []
        centroids_px = []
        logger.info("Start evolution")
        # Evolution loop
        strategy = params.get("strategy", "deterministic")
        for i in range(nb_iter):

            # take best "survive" nb of elements from current population
            if strategy == "deterministic":#just take the best "survive"
                indices = np.arange(0, min(survive, len(X)))
            elif strategy == "stochastic":#take "survive" in a stochastic way
                indices = np.arange(0, len(X))
                choose_nb = min(survive, len(X))
                temp = params.get("temperature", 1)
                prob = np.exp(-evals*temp)/np.exp(-evals*temp).sum()
                indices = np.random.choice(indices, size=choose_nb, replace=False, p=prob)
            elif strategy == "diversity": # take the best but make individuals only compete with individuals that are similar to them
                from sklearn.cluster import KMeans
                choose_nb = min(survive, len(X))
                n_clusters = params.get("n_clusters", 3)
                assert choose_nb % n_clusters == 0, "{} is not divisible by {}".format(choose_nb, n_clusters)
                take_per_cluster = choose_nb / n_clusters                
                clus = KMeans(n_clusters=n_clusters, n_init=50)
                F = vect(feat)
                clus.fit(F)
                clusid = clus.predict(F)
                all_indices = np.arange(0, len(X))
                indices = []
                for i in range(n_clusters):
                    indclus = all_indices[clusid==i]
                    indclus = sorted(indclus, key=lambda ind:evals[ind])
                    indclus = indclus[0:take_per_cluster]
                    indices.extend(indclus)
                indices = np.array(indices)
            else:
                raise Exception("Unknown strategy : {}".format(strategy))
            best = X[indices]
            best_evals = evals[indices]
            best_feat = g(best)
            archive_best_indices = archive_popul_indices[indices] # update best indices in archive

            # generate children
            if params.get("flatten", False):
                shape = best_feat.shape[1:]
                best_feat_flat = best_feat.reshape((best_feat.shape[0], -1))
                children_feat_flat = apply_genetic(best_feat_flat, nb=nb)
                children_feat = children_feat_flat.reshape((children_feat_flat.shape[0],) + shape)
            else:
                children_feat = apply_genetic(best_feat, nb=nb)

            # Evaluate children
            if layer_name == "input":
                if params.get("recons", False) == False:
                    children_px = children_feat
                else:
                    children_px = f(children_feat)
            else:
                children_px = f(children_feat)
            children_evals = compute_fitness(children_px,
                                             feat=children_feat,
                                             orig=best,
                                             orig_feat=best_feat)
            # add centroid of children
            C = children_feat.mean(axis=0)[None, :, :, :]
            centroids.append(C)
            
            C_px = f(C)
            centroids_px.append(C_px)

            # update archive with children
            archive_px = np.concatenate((archive_px, children_px), axis=0)
            archive_feat = np.concatenate((archive_feat, children_feat), axis=0)
            archive_evals = np.concatenate((archive_evals, children_evals), axis=0)
            a = len(archive_px) - len(children_px)
            # children indices on archive are added
            archive_children_indices = np.arange(a, a + len(children_px))
            # Now The current population = best + children, sort it according to eval
            evals = np.concatenate((best_evals, children_evals), axis=0)
            X = np.concatenate((best, children_px), axis=0)
            feat = np.concatenate((best_feat, children_feat), axis=0)
            archive_popul_indices = np.concatenate((archive_best_indices, archive_children_indices), axis=0)
            indices = np.argsort(evals)
            X, evals, feat = X[indices], evals[indices], feat[indices]
            archive_popul_indices = archive_popul_indices[indices]
            logger.info("Population mean Fitness : {}".format(evals.mean()))

        print(evals[0:10])
        # t-sne
        t_sne = params.get("tsne", False)
        if t_sne:
            logger.info("t-sne all the generated samples...")
            from sklearn.manifold import TSNE
            sne = TSNE(verbose=1, n_components=2, **params.get("tsneparams", {}))
            if params.get("tsnecentroids", True):
                centroids_ = vect(np.concatenate(centroids, axis=0))

                F = vect(archive_feat)
                F = np.concatenate((centroids_, F[0:nb_initial].mean(axis=0, keepdims=True), F[archive_popul_indices]), axis=0)
                first, last = 0, len(centroids_)
                inter = np.arange(first, last)
                first = last
                last += 1
                initial = np.arange(first, last)
                first = last
                last += len(archive_popul_indices)
                final = np.arange(first, last)
            else:
                F = vect(archive_feat)
                inter = np.arange(0, len(F))
                initial = np.arange(0, nb_initial)
                final = archive_popul_indices
            F_ = sne.fit_transform(F)
            fig = plt.figure()
            plt.scatter(F_[inter, 0], F_[inter, 1],
                        c=np.arange(len(inter)),
                        cmap='YlGn', label="intermediary")
            plt.scatter(F_[initial, 0], F_[initial, 1], c='yellow', label="initial population")
            plt.scatter(F_[final, 0], F_[final, 1], c='red', label="final population")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       fancybox=True, shadow=True, ncol=5)
            plt.savefig(params.get("tsnefile", "tsne.png"))
            plt.close(fig)
        
        # image scatter
        image_scatter = params.get("image_scatter", False)
        if image_scatter:
            logger.info("Scattering all images")
            from sklearn.manifold import TSNE
            from image_scatter import image_scatter
            from skimage.io import imsave
            import seaborn as sns
            
            if params.get("tsnecentroids", True):
                print(centroids[0].shape)

                initial = np.arange(0, nb_initial)
                initial_feat = vect(archive_feat[initial])
                initial_px = archive_px[initial]

                centroids_feat = vect(np.concatenate(centroids, axis=0))

                F = np.concatenate((initial_feat, centroids_feat), axis=0)
                px = np.concatenate((initial_px,) + tuple(centroids_px), axis=0)
            else:
                F = vect(archive_feat, only_max=True)
                px = archive_px
            #time = np.arange(len(F)).astype(np.float32) / len(F)
            #F = np.concatenate((F, time[:, None]), axis=1)

            from sklearn.decomposition import PCA
            #sne = TSNE(verbose=1, n_components=2, **params.get("tsneparams", {}))
            sne = PCA(n_components=2)
            F_ = sne.fit_transform(F)

            inter = np.arange(0, len(F))
            initial = np.arange(0, nb_initial)
            final = archive_popul_indices
            if c == 1:
                
                border = 0
                nbcols = len(px)
                gradient = np.array(sns.color_palette("hot", nbcols))

                shape = (px.shape[0], w + border, h, 3)
                px_ = np.zeros(shape).astype(np.float32)
                px_[:, :, :, :] = gradient[:, None, None, :]
                px_[:, 0:w, :] *= px[:, 0, :, :].reshape((px.shape[0], w, h, 1))  
                px_[:, w:, :, :] = gradient[:, None, None, :]
                img_res = w * 3
            else:
                px_ = px.transpose((0, 2, 3, 1))
                img_res = w * 3
            img = image_scatter(F_, px_, img_res=img_res)
            #fig = plt.figure()
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig(params.get("image_scatter_out", "image_scatter.png"))
            #plt.close(fig)
            imsave(params.get("image_scatter_out", "image_scatter.png"), img)
        return X, feat, evals

    if just_get_function:
        return perform
    else:
        from lasagnekit.misc.plot_weights import grid_plot
        y, feat, evals = perform()

        logger.info("Group plotting...")
        fig = plt.figure()
        if y.shape[1] == 1:
            opt = {"cmap": "gray"}
            y = y[:, 0]
        else:
            opt = {}
            y = y.transpose((0, 2, 3, 1))
        grid_plot(y[0:params.get("groupshow", 100)],
                  imshow_options=opt,
                  fig=fig)
        plt.savefig(out)
        save_all = params.get("save_all", False)
        folder = params.get("save_all_folder", "out")
        if save_all:
            logger.info("Save all individual images...")
            from skimage.io import imsave
            import pandas as pd
            import json
            for idx, sample in enumerate(y):
                sample -= sample.min()
                sample /= sample.max()
                filename = "{}/{}.png".format(folder, idx)
                logger.info("saving {}...".format(filename))
                imsave(filename, sample)
            logger.info("Save evals...")
            pd.Series(evals).to_csv("{}/evals.csv".format(folder))
            logger.info("Save params...")
            json.dump(params, open("{}/params.json".format(folder), "w"))
            logger.info("Save representations...")
            np.savez(open("{}/repr_{}.pkl".format(folder, layer_name), "w"), feat=feat)

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

def mutation(A, born_perc=0.1, dead_perc=0.1, nbtimes=1, val=10, inplace=True, nb=100):
    perc = born_perc + dead_perc
    nb_filters = A.shape[1]
    size = int(perc * nb_filters)

    if inplace:
        A_new = A
    else:
        replace = True if nb > A.shape[0] else False
        indices = np.random.choice(range(len(A)), size=nb, replace=replace)
        A_new = A[indices].copy()
    for i in range(nbtimes):
        for a in A_new:
            indices = np.random.choice(np.arange(len(a)),
                                       size=size, replace=True)
            nb_born = int(born_perc*nb_filters)
            born_indices = indices[0:nb_born]
            dead_indices = indices[nb_born:]
            a[dead_indices] = 0
            for idx in born_indices:
                a[idx] = 0
                if len(a.shape) == 3:
                    x, y = np.random.randint(a.shape[1]), np.random.randint(a.shape[2])
                    a[idx, x, y] = val
                elif len(a.shape) == 2:
                    x = np.random.randint(a.shape[1])
                    a[idx, x] = val
                elif len(a.shape) == 1:
                    a[idx] = val
                else:
                    raise Exception("WTF are you giving here?")
    return A_new

def switcher(A, nbtimes=1, size=2, nb=100, inplace=True):
    
    if len(A.shape) <= 2:
        return A

    if inplace:
        A_new = A
    else:
        replace = True if nb > A.shape[0] else False
        indices = np.random.choice(range(len(A)), size=nb, replace=replace)
        A_new = A[indices].copy()

    for _ in range(nbtimes):
        for a in A_new:
            indices = np.random.choice(len(a), size=size, replace=True)
            indices_shuffled = indices.copy()
            np.random.shuffle(indices_shuffled)
            for i, j in zip(indices, indices_shuffled):
                xi, yi = np.unravel_index(a[i].argmax(), a[i].shape)
                xj, yj = np.unravel_index(a[j].argmax(), a[j].shape)
                a[i, xi, yi], a[i, xj, yj] = a[i, xj, yj], a[i, xi, yi]
                a[j, xi, yi], a[j, xj, yj] = a[j, xj, yj], a[j, xi, yi]
    return A_new

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

"""
def multigenetic(capsule, data, layers, w, h, c,
                 **params):
    print(layers.keys())
    layer_names = params.get("layer_names", "wta_spatial")
    names = layer_names
    layerval = {name: T.tensor4() for name in names}
    layervalordered = [layerval[name] for name in names]

    nb_iter = params.get("nb_iter", 100)
    just_get_function = params.get("just_get_function", False)
    out = params.get("out", "out.png")
    
    logger.info("Compiling functions...")
    x = T.tensor4()
    g = theano.function(
        [x],
        [L.get_output(layers[name], x) for name in names]
    )
    recons = theano.function(
        layervalordered,
        L.get_output(layers["output"], {layers[name]: layerval[name] for name in names}))

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

    def smarter_mutation(A, born_perc=0.1, dead_perc=0.1, nbtimes=1, val=10, nb=100):
        # val = A.max()
        perc = born_perc + dead_perc
        nb_filters = A.shape[1]
        size = int(perc * nb_filters)
        for i in range(nbtimes):
            for a in A:
                indices = np.random.choice(np.arange(a.shape[1]),#was a  bug in the old genetic
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

    def vect(F):
        F = [f.max(axis=(2, 3)).reshape((f.shape[0], -1)) for f in F]
        return np.concatenate(F, axis=1)
      
    def nearestneighbours_distance(X, orig=None, feat=None, orig_feat=None):
        assert feat is not None
        from sklearn.neighbors import NearestNeighbors
        K = params.get("nearest_neighbors", 8)
        X_ = vect(feat)
        if orig_feat is not None:
            O_ = vect(orig_feat)
            S = np.concatenate((X_, O_), axis=0)
        else:
            S = X_
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(S)
        distances, _ = nbrs.kneighbors(S)
        if orig is not None:
            D = distances[0:len(X_), :]
        else:
            D = distances
        return D.mean(axis=1)

    def diversity(X, orig=None, feat=None, orig_feat=None):
        return -nearestneighbours_distance(
                X, orig=orig,
                feat=feat,
                orig_feat=orig_feat)

    rec_model = None
    rec_class = None

    def recognizability(X, feat=None, orig=None, orig_feat=None):
        # maximize prediction of some category
        pred = rec_model.predict_proba(X)[:, rec_class]
        return 1. - pred

    def reconstruction(X, feat=None, orig=None, orig_feat=None):
        return ((X - capsule.reconstruct(X)) ** 2).sum(axis=(1, 2, 3))

    def reconstruction_and_diversity(X, feat=None, orig=None, orig_feat=None):
        diversity = nearestneighbours_distance(X, feat=feat, orig=orig,
                                               orig_feat=orig_feat)
        # minimize reconstruction and maximize  diversity
        return reconstruction(X, orig=orig) - params.get("tradeoff", 0.01) * diversity
    # Choose and init fitness
    logger.info("Init fitness...")
    
    fitness_name = params.get("fitness_name", "reconstruction")
    fitness = {
        "reconstruction": reconstruction,
        "reconstruction_and_diversity": reconstruction_and_diversity,
        "recognizability": recognizability,
        "diversity": diversity
    }
    compute_fitness = fitness[fitness_name]

    if compute_fitness == recognizability:
        from keras.models import model_from_json
        arch = params.get("arch", "models/mnist.json")
        modelfile = params.get("modelfile", "models/mnist.hdf5")
        rec_model = model_from_json(open(arch).read())
        rec_model.load_weights(modelfile)
        rec_class = params.get("category", 3)

    def perform():

        # Init data
        nb_initial = params.get("nb_initial", data.X.shape[0])
        initial_source = params.get("initial_source", "random")
    
        # nb_initial = 10       
        if initial_source == "dataset":
            X = data.X.reshape((data.X.shape[0], c, w, h))
            X = X[0:nb_initial]
        elif initial_source == "random":
            X = np.random.uniform(size=(data.X.shape[0], c, w, h))
            X = X[0:nb_initial]
        elif initial_source == "centroids":
            categories = list(set(data.y))
            print(categories)
            centroid = np.zeros((len(categories), c, w, h))
            centroid_size = [0] * len(categories)
            for i in range(10):
                data.load()
                X = data.X.reshape((data.X.shape[0], c, w, h))
                for idx, cat in enumerate(categories):
                    S = X[data.y==cat]
                    centroid[idx] += S.sum(axis=0)
                    centroid_size[idx] += len(S)

            for ctroid, ctroidsz in zip(centroid, centroid_size):
                print(ctroidsz, ctroid.max())
                ctroid /= ctroidsz

            X = np.array(centroid).astype(np.float32)
            nb_initial = X.shape[0]
        else:
            raise Exception("bad initial")

        X = X.astype(np.float32)

        # genetic params
        nb = params.get("nbchildren", 100)  # nb of children per iteration
        survive = params.get("nbsurvive", 20)

        # Init genetic
        logger.info("Compute fitness of initial population...")
        feat = g(X)
        evals = compute_fitness(X, feat=feat)
        indices = np.argsort(evals)
        X, evals, feat = X[indices], evals[indices], [f[indices] for f in feat]
        print(evals[0:10])

        archive_px = X.copy()  # all generated images in pixel space
        archive_feat = [f.copy() for f in feat]  # all generated images in feature space
        archive_evals = evals.copy()  # all evaluations of genereted images
        archive_popul_indices = indices  # current indices on archive of the population
        centroids = []
        logger.info("Start evolution")
        # Evolution loop
        strategy = params.get("strategy", "deterministic")
        for i in range(nb_iter):

            # take best "survive" nb of elements from current population
            if strategy == "deterministic":
                indices = np.arange(0, min(survive, len(X)))
            elif strategy == "stochastic":
                indices = np.arange(0, len(X))
                choose_nb = min(survive, len(X))
                temp = params.get("temperature", 1)
                prob = np.exp(-evals*temp)/np.exp(-evals*temp).sum()
                indices = np.random.choice(indices, size=choose_nb, replace=False, p=prob)
            else:
                raise Exception("Unknown strategy : {}".format(strategy))
            best = X[indices]
            best_evals = evals[indices]
            best_feat = g(best)
            archive_best_indices = archive_popul_indices[indices] # update best indices in archive

            # generate children
            children_feat = [crossover(f, nb=nb) for f in best_feat]
            children_feat = [switcher(f) for f in children_feat]

            children_feat = [smarter_mutation(
                    f,
                    born_perc=params.get("born_perc", 0.1),
                    dead_perc=params.get("dead_perc", 0.1),
                    nbtimes=params.get("mutationnbtimes", 1),
                    val=params.get("mutationval", 10)
            ) for f in children_feat]
            centroids.append(vect(children_feat).mean(axis=0).tolist())
            children_px = recons(*children_feat)
            children_evals = compute_fitness(children_px,
                                             feat=children_feat,
                                             orig=best,
                                             orig_feat=best_feat)
            # update archive with children
            archive_px = np.concatenate((archive_px, children_px), axis=0)
            archive_feat = [np.concatenate((ar, ch), axis=0) for ar, ch in zip(archive_feat, children_feat)]
            archive_evals = np.concatenate((archive_evals, children_evals), axis=0)
            a = len(archive_px) - len(children_px)
            # children indices on archive are added
            archive_children_indices = np.arange(a, a + len(children_px))
            # Now The current population = best + children, sort it according to eval
            evals = np.concatenate((best_evals, children_evals), axis=0)
            X = np.concatenate((best, children_px), axis=0)
            feat = [np.concatenate((bst, ch), axis=0) for ch, bst in zip(children_feat, best_feat)]
            archive_popul_indices = np.concatenate((archive_best_indices, archive_children_indices), axis=0)
            indices = np.argsort(evals)
            X, evals, feat = X[indices], evals[indices], [f[indices] for f in feat]
            archive_popul_indices = archive_popul_indices[indices]
            logger.info("Population mean Fitness : {}".format(evals.mean()))

        print(evals[0:10])
        # t-sne
        t_sne = params.get("tsne", False)
        if t_sne:
            logger.info("t-sne all the generated samples...")
            from sklearn.manifold import TSNE
            sne = TSNE()
            if params.get("tsnecentroids", True):
                F = vect(archive_feat)
                F = np.concatenate((centroids, F[0:nb_initial].mean(axis=0, keepdims=True), F[archive_popul_indices]), axis=0)
                first, last = 0, len(centroids)
                inter = np.arange(first, last)
                first = last
                last += 1
                initial = np.arange(first, last)
                first = last
                last += len(archive_popul_indices)
                final = np.arange(first, last)
            else:
                F = vect(archive_feat)
                inter = np.arange(0, len(F))
                initial = np.arange(0, nb_initial)
                final = archive_popul_indices
            F_ = sne.fit_transform(F)
            fig = plt.figure()
            plt.scatter(F_[inter, 0], F_[inter, 1],
                        c=np.arange(len(inter)),
                        cmap='YlGn', label="intermediary")
            plt.scatter(F_[initial, 0], F_[initial, 1], c='yellow', label="initial population")
            plt.scatter(F_[final, 0], F_[final, 1], c='red', label="final population")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       fancybox=True, shadow=True, ncol=5)
            plt.savefig(params.get("tsnefile", "tsne.png"))
            plt.close(fig)
        return X, feat, evals

    if just_get_function:
        return perform
    else:
        from lasagnekit.misc.plot_weights import grid_plot
        y, feat, evals = perform()

        logger.info("Group plotting...")
        fig = plt.figure()
        if y.shape[1] == 1:
            opt = {"cmap": "gray"}
            y = y[:, 0]
        else:
            opt = {}
            y = y.transpose((0, 2, 3, 1))
        grid_plot(y[0:params.get("groupshow", 100)],
                  imshow_options=opt,
                  fig=fig)
        plt.savefig(out)
        save_all = params.get("save_all", False)
        folder = params.get("save_all_folder", "out")
        if save_all:
            logger.info("Save all individual images...")
            from skimage.io import imsave
            import pandas as pd
            import json
            for idx, sample in enumerate(y):
                sample -= sample.min()
                sample /= sample.max()
                filename = "{}/{}.png".format(folder, idx)
                logger.info("saving {}...".format(filename))
                imsave(filename, sample)
            logger.info("Save evals...")
            pd.Series(evals).to_csv("{}/evals.csv".format(folder))
            logger.info("Save params...")
            json.dump(params, open("{}/params.json".format(folder), "w"))
            logger.info("Save representations...")
            for ft, layer_name in zip(feat, layer_names):
                np.savez(open("{}/repr_{}.pkl".format(folder, layer_name), "w"), feat=ft)
"""


def recons(capsule, data, layers, w, h, c,
           layer_name="wta_spatial",
           **kw):
    from lasagnekit.misc.plot_weights import grid_plot
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

def denoising(capsule, data, layers, w, h, c,
              **params):
    from lasagnekit.misc.plot_weights import grid_plot

    save_each = params.get("save_each", 10)

    x = T.tensor4()
    nb_examples = params.get("nb_examples", 100)
    shape = (nb_examples,) + layers["input"].output_shape[1:]
    x = np.random.uniform(size=shape).astype(np.float32)
    prob = params.get("prob", 0.1)
    logger.info("Starting:")
    for i in range(params.get("nb_iter", 10)):
        logger.info("Iteration {}..".format(i))
        x *= (np.random.uniform(size=x.shape) <= prob)
        x = capsule.reconstruct(x)

        if i % save_each == 0:
            fig = plt.figure()
            grid_plot(x[:, 0], imshow_options={"cmap": "gray"})
            plt.savefig(params.get("out", "out.png"))
            plt.savefig("out.png")
            plt.close(fig)

def recons_from_features(capsule, data, layers, w, h, c,
                         layer_name="wta_spatial", out="out.png", **kw):
    from lasagnekit.misc.plot_weights import grid_plot
    from scipy.optimize import fmin_l_bfgs_b
    print(layers)
    name = layer_name
    logger.info("Compiling functions...")
    x = T.tensor4()
    CST = 15
    recons = L.get_output(layers["output"], {layers[name]: x * CST})
    get_layer = theano.function(
        [x],
        L.get_output(layers[name], x)
    )
    get_recons = theano.function(
        [x],
        recons
    )
    recons_error = ((recons - L.get_output(layers["output"], recons)) ** 2).sum(axis=(1, 2, 3)).mean()
    # norm = (x**2).sum()
    loss = recons_error # - 0.001 * norm
    get_loss = theano.function(
            [x],
            loss
    )

    get_loss = theano.function([x], loss)
    get_grad = theano.function([x], theano.grad(loss, x))

    nb_examples = 100
    shape_inp = (nb_examples,) + (c, w, h)
    shape = (nb_examples,) + layers[name].output_shape[1:]

    def eval_loss(x0):
        x0_ = x0.reshape(shape).astype(np.float32)
        l = get_loss(x0_)
        return l

    def eval_grad(x0):
        x0_ = x0.reshape(shape).astype(np.float32)
        g = np.array(get_grad(x0_)).flatten().astype(np.float64)
        return g
    logger.info("Setting initial population...")
    x = get_layer(np.random.uniform(size=shape_inp).astype(np.float32))
    x = x.astype(np.float32)
    shape = x.shape
    logger.info("Start gradient descent!")
    for i in range(15):
        x, _, _ = fmin_l_bfgs_b(eval_loss, x.flatten(), fprime=eval_grad, maxfun=40)
        x = x.reshape(shape)
        x = x.astype(np.float32)
        #g = get_grad(x)
        #x -= 0.5 * np.array(g)
        print(get_loss(x))
        if i % 10 == 0:
            y = get_recons(x)
            fig = plt.figure()
            if y.shape[1] == 1:
                opt = {"cmap": "gray"}
                y = y[:, 0]
            else:
                opt = {}
                y = y.transpose((0, 2, 3, 1))
            grid_plot(y,
                      imshow_options=opt,
                      fig=fig)
            plt.savefig(out)


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


def notebook(capsule, data, layers, w, h, c, **kw):
    return capsule, data, layers, w, h, c


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
