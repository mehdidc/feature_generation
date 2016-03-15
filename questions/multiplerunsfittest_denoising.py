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
	parser.add_argument('--prob', default=0.5, type=int)

	args = parser.parse_args()

	all_params = []
	fitness_name = args.fitness_name
	layer_name = args.layer_name
	print(layer_name)
	if args.val is None:
		val = {"input": 2, "conv1": 300, "conv2": 200, "conv3": 10, "wta_spatial": 10, "wta_channel": 10}[layer_name]
	else:
		val = args.val
	prob = args.prob
	nb_runs = args.nb_runs
	initial_source = args.initial_source


	for i in range(nb_runs):
		seed = np.random.randint(0,999999999)
		print(seed)
		name = "exp{}_{}".format(fitness_name, val)
		folder = "answers/multiplerunsfittest_denoising/{}_{}_{}_{}_{}/run{}".format(fitness_name, layer_name, val, initial_source, prob, i + 1)
		mkdir_path(folder)
		out = folder + "/out"
		params = {
			"seed": seed,
			"nb_iter": nb_iter,
			"nb_examples": nb_examples,
			"prob": prob,
			"val": val,
			"out": out,
		}
		all_params.append(params)
	check(filename="models/model_E.pkl",
          what="genetic",
          dataset="digits",
          params=all_params)
