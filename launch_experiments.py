import uuid
from tempfile import NamedTemporaryFile
import json
import subprocess


def summarize(s):
    return str(uuid.uuid4())


def launch(model_name="model8", dataset="digits", params=None, prefix=None):
    if params is None:
        params = {}
    if prefix is None:
        prefix = "training/default_folder/{}_{}".format(model_name, summarize(params))

    def random_name():
        return str(uuid.uuid4())

    name = "tmpjs/{}.json".format(random_name())
    fd = open(name, "w")
    json.dump(params, fd)
    fd.close()
    cmd = "sbatch ./launch invoke train --dataset={} --model-name={} --prefix={} --params={}"
    cmd = cmd.format(
        dataset,
        model_name,
        prefix,
        name,
    )
    print(cmd)
    subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    params = dict(
        model_params=dict(
            nb_filters=[64, 64, 64, 128, 128, 128, 256, 256, 256],
            filter_size=3,
            use_wta_channel=True,
            use_wta_spatial=True,
            nb_filters_mul=1,
            wta_channel_stride=2,
            nb_layers=8)
    )
    launch(model_name="model55", dataset="digits", params=params)
