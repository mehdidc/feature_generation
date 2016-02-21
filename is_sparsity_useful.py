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
			"fitness_name": "reconstruction",
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