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
	val_range = (0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 7, 10)
	#val_range = (10, 10.5, 11, 11.5, 12, 15, 20)
	fitness_name_range = ("reconstruction",)
	#flatten_range = (True, False)

	for fitness_name, val in product(fitness_name_range, val_range):		
		name = "exp{}_{}".format(fitness_name, val)
		folder = "answers/paperconv/{}".format(name)		
		mkdir_path(folder)
		params = {
			"save_all": True,
			"save_all_folder": folder,
			"nb_iter": 100,
			"tsne": True,
			"tsnefile": "{}/tsne.png".format(folder),
			"out": "{}/out".format(folder),

			"layer_name": "conv3",
			"nb_iter": 100,
			"fitness_name": "reconstruction",
			"initial_source": "random",

			"evalsfile": "{}/evals.csv".format(folder),
			"evalsmeanfile": "{}/evalsmean.csv".format(folder),

			#nb_initial": 100,

			"nbchildren": 100,
			"nbsurvive": 20,

			"strategy": "replace_worst",
			"born_perc": 0.1,
			"dead_perc": 0.1,
			"mutationval": val,
			"nbtimes": 1,
			"seed": 1,

			"flatten": False,
			"recons": True

		}
		all_params.append(params)
	check(filename="models/model_E.pkl",
          what="genetic",
          dataset="digits",
          params=all_params)