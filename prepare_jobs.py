import uuid
from tempfile import NamedTemporaryFile
import json
import subprocess
import hashlib
from collections import OrderedDict
import os

def summarize(d):
    s = json.dumps(d, sort_keys=True)
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()


def build_cmd(model_name="model8", dataset="digits", params=None, prefix=None, budget_hours=None):
    summarized_name = summarize(params)
    if params is None:
        params = {}
    if prefix is None:
        prefix = "jobs/results/{}_{}".format(model_name, summarized_name)
    name = "jobs/params/{}_{}.json".format(model_name, summarized_name)
    fd = open(name, "w")

    json.dump(params, fd)
    fd.close()
    output = "jobs/outputs/{}".format(summarized_name)
    extra = ""
    if budget_hours is not None:
        extra = "--budget-hours={}".format(budget_hours)
    cmd = "sbatch --output={} --error={} --mail-user=mcherti --mail-type=FAIL --mail-type=END ./launch invoke train --dataset={} --model-name={} --prefix={} --params={} {}"
    cmd = cmd.format(
        output,
        output,
        dataset,
        model_name,
        prefix,
        name,
        extra
    )
    return cmd


def job_write(id_, cmd, where=""):
    name_finished = "jobs/finished/{}".format(id_)
    if os.path.exists(name_finished):
        print("Job {} already done".format(id_))
        return
    name = "jobs/available/{}/{}".format(where, id_)
    with open(name, "w")  as fd:
        fd.write(cmd)

if __name__ == "__main__":

    def build_params(model_params,
                     denoise,
                     noise,
                     walkback,
                     walkback_jump,
                     autoencoding_loss,
                     contractive,
                     contractive_coef,
                     marginalized,
                     binarize_thresh):
        params = OrderedDict(
            model_params=model_params,
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_jump=walkback_jump,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh,
        )
        return params

    def jobset1():

        nb_filters_per_layer = {
            1: [256],
            2: [128, 256],
            3: [64, 128, 256],
            4: [64, 128, 128, 256],
            5: [64, 64, 128, 128, 256],
        }

        def build_model_params(nb_filters,
                               nb_layers,
                               filter_size,
                               use_wta_spatial, use_wta_channel,
                               nb_filters_mul,
                               wta_channel_stride):
            return OrderedDict(nb_filters=nb_filters,
                               filter_size=filter_size,
                               use_wta_channel=use_wta_channel,
                               use_wta_spatial=use_wta_spatial,
                               nb_filters_mul=nb_filters_mul,
                               wta_channel_stride=wta_channel_stride,
                               nb_layers=nb_layers)

        all_params = (
            build_params(
                build_model_params(nb_filters_per_layer[nb_layers],
                                   nb_layers, filter_size,
                                   use_wta_spatial, use_wta_channel,
                                   nb_filters_mul,
                                   wta_channel_stride),
                denoise,
                noise,
                walkback,
                walkback_jump,
                autoencoding_loss,
                contractive,
                contractive_coef,
                marginalized,
                binarize_thresh)
                for nb_layers in (2, 3, 4, 5)
                for filter_size in (3, 5)
                for use_wta_spatial in (True, False)
                for use_wta_channel in (True, False)
                if use_wta_spatial is True or use_wta_channel is True
                for nb_filters_mul in (1,)
                for wta_channel_stride in ((2, 4) if use_wta_channel else (1,))
                for denoise in (None, 0.5)
                for noise in ("zero_masking",)
                for walkback in ((1, 3, 5) if denoise is not None else (1,))
                for walkback_jump in ((True, False) if walkback is True else (False,))
                for autoencoding_loss in ("squared_error",)
                for contractive in (False,)
                for contractive_coef in (0,)
                for marginalized in (False,)
                for binarize_thresh in (None, 0.5)
        )
        all_params = list(all_params)
        dataset = "digits"
        model_name = "model55"
        for p in all_params:
            s = summarize(p)
            cmd = build_cmd(model_name="model55", dataset="digits", params=p)
            job_write(s, cmd)

    def jobset2():

        def build_model_params(nb_layers,
                               use_wta_lifetime,
                               wta_lifetime_perc,
                               nb_hidden_units):
            return OrderedDict(nb_layers=nb_layers,
                               use_wta_lifetime=use_wta_lifetime,
                               wta_lifetime_perc=wta_lifetime_perc,
                               nb_hidden_units=nb_hidden_units)

        all_params = (
            build_params(
                build_model_params(nb_layers,
                                   use_wta_lifetime,
                                   wta_lifetime_perc,
                                   nb_hidden_units),
                denoise,
                noise,
                walkback,
                walkback_jump,
                autoencoding_loss,
                contractive,
                contractive_coef,
                marginalized,
                binarize_thresh)
                for nb_hidden_units in (500, 1000, 2000)
                for nb_layers in (1, 2, 3)
                for use_wta_lifetime in (True, False)
                for wta_lifetime_perc in ((0.02, 0.05, 0.1, 0.15) if use_wta_lifetime else (None,))
                for denoise in (None, 0.5)
                for noise in ("zero_masking",)
                for walkback in ((1, 3, 5) if denoise is not None else (1,))
                for walkback_jump in ((True, False) if walkback is True else (False,))
                for autoencoding_loss in ("squared_error",)
                for contractive in (False,)
                for contractive_coef in (0,)
                for marginalized in (False,)
                for binarize_thresh in (None, 0.5)
        )
        all_params = list(all_params)
        print(len(all_params))
        for p in all_params:
            s = summarize(p)
            cmd = build_cmd(model_name="model56", dataset="digits", params=p, budget_hours=24)
            print(cmd)
            job_write(s, cmd, where="jobset2")
    #jobset1()
    jobset2()
