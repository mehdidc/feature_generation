if __name__ == "__main__":
	from tasks import check
	import random
	import uuid
	import os
	all_params = []
	for i in range(1000):
		folder = "iccc/{}".format(str(uuid.uuid4()))
		try:
			os.mkdir(folder)
		except Exception:
			pass
		params = {
			"save_all": True,
			"save_all_folder": folder,
			"nb_iter": 100,
			"tsne": True,
			"tsnefile": "{}/tsne.png".format(folder),
			"out": "{}/out.png".format(folder),

			"layer_name": random.choice(("conv1", "conv2", "conv3", "wta_spatial", "wta_channel")),
			"nb_iter": 100,
			"fitness_name": "reconstruction_and_diversity",
			"tradeoff": random.choice((0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)), #tradeoff reconstruction/diversity if fitness_name is reconstruction_and_diversity
			"nearest_neighbors": random.choice((3, 5, 8, 10)),#diversity nearest neighbors
			"initial_source": random.choice(("random", "dataset")),
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