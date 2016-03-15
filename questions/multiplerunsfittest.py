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
	parser.add_argument('--val', default=None, type=int)
	parser.add_argument('--cv', default=None, type=int)

	args = parser.parse_args()

	all_params = []
	fitness_name = args.fitness_name
	layer_name = args.layer_name
	print(layer_name)
	if args.val is None:
		val = {"input": 2, "conv1": 300, "conv2": 200, "conv3": 10, "wta_spatial": 10, "wta_channel": 10}[layer_name]
	else:
		val = args.val
	nb_runs = args.nb_runs
	initial_source = args.initial_source


	for i in range(nb_runs):
		seed = np.random.randint(0,999999999)
		print(seed)
		name = "exp{}_{}".format(fitness_name, val)
		folder = "answers/multiplerunsfittest/{}_{}_{}_{}/run{}".format(fitness_name, layer_name, val, initial_source, i + 1)
		mkdir_path(folder)
		params = {
			"save_all": True,
			"save_all_folder": folder,

			"tsne": True,
			"tsnefile": "{}/tsne.png".format(folder),
			"out": "{}/out".format(folder),
			"evalsfile": "{}/evals.csv".format(folder),
			"evalsmeanfile": "{}/evalsmean.csv".format(folder),

			"layer_name": layer_name,
			"nb_iter": 150,
			"fitness_name": fitness_name,
			"initial_source": initial_source,
			#nb_initial": 100,

			"nbchildren": 100,
			"nbsurvive": 20,

			"strategy": "determnistic_only_children",
			"dead_perc": 0.1,
			"mutationval": val,
			"nbtimes": 1,
			"seed": seed,

			"flatten": True if layer_name == "input" else False,
			"recons": True,
			"group_plot_save": [],
		}
		all_params.append(params)
	check(filename="models/model_E.pkl",
          what="genetic",
          dataset="digits",
          params=all_params)
