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
	tradeoff_range = (0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10, 20)
	nearest_neighbors_range = (3, 5, 8, 10)
	
	layer_name_range = ("conv1", "conv2", "conv3", "wta_spatial", "wta_channel")

	for nearest_neighbors, layer_name, tradeoff in product(nearest_neighbors_range, layer_name_range, tradeoff_range):		
		name = "exp{}_{}_{}".format(tradeoff, nearest_neighbors, layer_name)
		folder = "answers/tradeoff_reconstruction_diversity/{}".format(name)		
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
			"fitness_name": "reconstruction_and_diversity",
			"tradeoff": tradeoff, #tradeoff reconstruction/diversity if fitness_name is reconstruction_and_diversity
			"nearest_neighbors": nearest_neighbors,#diversity nearest neighbors
			"initial_source": "random",
			"nb_initial": 100,

			"nbchildren": 100,
			"survive": 20,
			"strategy": "deterministic",
			"born_perc": 0.1,
			"dead_perc": 0.1,
			"mutationval": 10,
			"nbtimes": 1,
		}
		all_params.append(params)
	check(filename="models/model_E.pkl",
          what="genetic",
          dataset="digits",
          params=all_params)