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
	val_range = (0.01, 0.1, 0.5, 1, 5, 10, 20, 30, 50, 80, 100, 200,)
	fitness_range = ("reconstruction",
			         #"diversity",
			         #"reconstruction_and_diversity",
			         "pngsize",
			      	 "recognizability") 

	for fitness, val, layer_name in product(fitness_range, val_range, layer_name_range):
		name = "exp{}_{}_{}".format(layer_name, val, fitness)
		folder = "answers/comparing_fitness/{}".format(name)
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
			"fitness_name": fitness,
			"initial_source": "random",
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
	check(filename="models/model_E.pkl",
          what="genetic",
          dataset="digits",
          params=all_params)