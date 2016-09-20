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
	layer_name_range = ("conv1", "conv2", "conv3")
	initial_source_range = ("random", "dataset")
	val_range = (0.01, 0.1, 0.5, 1, 5, 10, 20, 30, 50, 80, 100, 200,)

	for layer_name, initial_source, val in product(layer_name_range, initial_source_range, val_range):
		name = "exp{}_{}_{}".format(layer_name, initial_source, val)
		folder = "answers/is_sparsity_useful/{}".format(name)
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
			"nb_initial": 100,

			"nbchildren": 100,
			"nbsurvive": 20,
			"strategy": "deterministic",
			"born_perc": 0.1,
			"dead_perc": 0.1,
			"mutationval": val,
			"nbtimes": 1,
		}
		all_params.append(params)
	check(filename="training/31mnist/model.pkl",
          what="genetic",
          dataset="digits",
          params=all_params)