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
	all_params = []
	for i in range(10):
		folder = "answers/is_sparsity_useful/{}".format(str(uuid.uuid4()))
		mkdir_path(folder)
		params = {
			"save_all": True,
			"save_all_folder": folder,
			"nb_iter": 100,
			"tsne": True,
			"tsnefile": "{}/tsne.png".format(folder),
			"out": "{}/out.png".format(folder),

			"layer_name": random.choice(("unconv0",)),
			"nb_iter": 100,
			"fitness_name": "reconstruction",
			"initial_source": random.choice(("random", "dataset")),
			"nb_initial": 100,

			"nbchildren": 100,
			"survive": 20,
			"strategy": "deterministic",
			"born_perc": 0.1,
			"dead_perc": 0.1,
			"mutationval": random.choice((0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)),
			"nbtimes": 1,
		}
		all_params.append(params)
	check(filename="models/18mnist.pkl",
          what="genetic",
          dataset="digits",
          params=all_params)