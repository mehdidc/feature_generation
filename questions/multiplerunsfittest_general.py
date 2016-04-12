import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tasks import check


def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)


if __name__ == "__main__":
    import random
    import uuid
    import os
    from itertools import product
    import numpy as np
    np.random.seed(2)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness_name', default='reconstruction', type=str)
    parser.add_argument('--layer_name', default='input', type=str)
    parser.add_argument('--nb_runs', default=100, type=int)
    parser.add_argument('--initial_source', default="random", type=str)
    parser.add_argument("--perc", default=0.1, type=float)
    parser.add_argument("--sel", default=1, type=int)
    parser.add_argument("--crossover", default=1, type=int)
    parser.add_argument("--mutation", default=1, type=int)
    parser.add_argument("--model", default="sparseautoencoder", type=str)
    parser.add_argument("--sort", default=1, type=int)
    model_to_filename = {
        "sparseautoencoder": "models/model_E.pkl",
        "nonsparseautoencoder": "training/31mnist/model.pkl",
        "nonsparsedenoisingautoencoder": "training/31digits_denoisev2/model.pkl",
        "sparseautoencoderfonts": "training/22fonts/model.pkl",
        "sparseautoencoderflaticons": "training/16flat/model.pkl",
        "denoisingautoencoder": "training/41digits_denoise/model.pkl",
        "sparsedenoisingautoencoder": "training/8digitsdenoise/model.pkl",
        "walkbackdenoisingautoencoder": "training/41digits_walkback/model.pkl"
    }
    args = parser.parse_args()
    sort = args.sort
    model_filename = model_to_filename[args.model]
    all_params = []
    fitness_name = args.fitness_name
    layer_name = args.layer_name
    print(layer_name)
    perc = args.perc
    nb_runs = args.nb_runs

    sel = args.sel
    cv = args.crossover
    mut = args.mutation
    initial_source = args.initial_source


    for i in range(nb_runs):
        seed = np.random.randint(0,999999999)
        print(seed)
        name = "exp{}".format(fitness_name)
        folder = "answers/{}_multipleruns/{}_{}_{}_{}_{}_{}_{}_{}/run{}".format(args.model,
                                                                                fitness_name,
                                                                                layer_name,
                                                                                initial_source,
                                                                               "sel" if sel==1 else "nosel",
                                                                               "cv" if cv==1 else "nocv",
                                                                               "mut" if mut==1 else "nomut",
                                                                                perc,
                                                                                "sort" if sort==1 else "nosort",
                                                                                i + 1)
        mkdir_path(folder)
        params = {
            "save_all": True,
            "save_all_folder": folder,

            "tsne": False,
            "tsnefile": "{}/tsne.png".format(folder),
            "out": "{}/out".format(folder),
            "evalsfile": "{}/evals.csv".format(folder),
            "evalsmeanfile": "{}/evalsmean.csv".format(folder),

            "layer_name": layer_name,
            "nb_iter": 100,
            "fitness_name": fitness_name,
            "initial_source": initial_source,

            "nb_initial": 100,

            "nbchildren": 100,
            "nbsurvive": 50,

            "strategy": "deterministic_only_children" if sel == 1 else "nosel",
            "dead_perc": perc,
            "nbtimes": 1,
            "seed": seed,
            "do_mutation": True if mut==1 else False,
            "do_crossover": True if cv==1 else False,
            "sort": True if sort==1 else False,
            "flatten": True if layer_name == "input" else False,
            "recons": True,
        }
        all_params.append(params)
    check(filename=model_filename,
            what="genetic",
            dataset="digits",
            params=all_params,
            force_w=28, force_h=28)
