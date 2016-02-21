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
	nb_initial_range = (10, 20, 40, 60, 100, 120, 200, 300)
	layer_name_range = ("conv3", "wta_spatial", "wta_channel")
	initial_source_range = ("dataset", "random", "centroids")
	already = set()
	for nb_initial, initial_source, layer_name in product(nb_initial_range, initial_source_range, layer_name_range):		
		if initial_source == "centroids":
			if layer_name in already:
				continue
			else:
				already.add(layer_name)
		name = "exp{}_{}_{}".format(nb_initial, initial_source, layer_name)
		folder = "answers/effect_of_initial/{}".format(name)		
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
			"initial_source": initial_source,
			"nb_initial": nb_initial,

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