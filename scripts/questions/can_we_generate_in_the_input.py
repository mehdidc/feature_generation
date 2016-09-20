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
	born_perc_range = (0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 0.9)
	dead_perc_range = (0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 0.9)
	#val_range = (0.01, 0.1, 0.5, 1, 5, 10, 20, 30, 50, 80, 100, 200,)
	val_range = (1, 5, 10, 20, 30, 50, 100)
	for born_perc, dead_perc, val in product(born_perc_range, dead_perc_range, val_range):
		name = "exp{}_{}_{}".format(born_perc, dead_perc, val)
		folder = "answers/geninput/{}".format(name)		
		mkdir_path(folder)
		params = {
			"save_all": True,
			"save_all_folder": folder,
			"nb_iter": 100,
			"tsne": True,
			"tsnefile": "{}/tsne.png".format(folder),
			"out": "{}/out.png".format(folder),

			"layer_name": "input",
			"nb_iter": 100,
			"fitness_name": "reconstruction",

			"nbchildren": 100,
			"nbsurvive": 20,
			"strategy": "deterministic",
			"born_perc": born_perc,
			"dead_perc": dead_perc,
			"mutationval": val,
			"nbtimes": 1,
			"recons": True,
			"flatten": True
		}
		all_params.append(params)
	check(filename="models/model_E.pkl",
          what="genetic",
          dataset="digits",
          params=all_params)