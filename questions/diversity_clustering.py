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
	all_params = []
	layer_name_range = ("conv1", "conv2", "conv3", "wta_spatial", "wta_channel")
	val_range = (0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 200,)
	n_clusters_range = (1, 2, 4, 10)
	for n_clusters, val, layer_name in product(n_clusters_range, val_range, layer_name_range):		
		name = "exp{}_{}_{}".format(n_clusters, val, layer_name, layer_name)
		folder = "answers/diversity_clustering/{}".format(name)		
		mkdir_path(folder)
		params = {
			"save_all": True,
			"save_all_folder": folder,
			"nb_iter": 100,
			"tsne": True,
			"tsnefile": "{}/tsne.png".format(folder),
			"out": "{}/out.png".format(folder),

			"layer_name": layer_name,
			"nb_iter": 100,
			"fitness_name": "reconstruction",
			"initial_source": "random",
			"nb_initial": 100,

			"nbchildren": 100,
			"nbsurvive": 20,
			"strategy": "diversity",
			"n_clusters": n_clusters,
			"born_perc": 0.1,
			"dead_perc": 0.1,
			"mutationval": val,
			"nbtimes": 1,
		}
		all_params.append(params)
	check(filename="models/model_E.pkl",
          what="genetic",
          dataset="digits",
          params=all_params)