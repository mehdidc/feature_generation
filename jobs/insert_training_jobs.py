import uuid
from tempfile import NamedTemporaryFile
import json
import subprocess
import hashlib
from collections import OrderedDict
import os

from lightjob.utils import summarize

def test():
        all_params = (
            build_params(
                OrderedDict(tied=tied,
                            use_wta_lifetime=use_wta_lifetime,
                            wta_lifetime_perc=wta_lifetime_perc,
                            nb_hidden_units=nb_hidden_units),
                denoise,
                noise,
                walkback,
                walkback_jump,
                autoencoding_loss,
                contractive,
                contractive_coef,
                marginalized,
                binarize_thresh)
            for nb_hidden_units in (250, 500)
            for use_wta_lifetime in (True,)
            for wta_lifetime_perc in (0.02,)
            for denoise in (None,)
            for noise in ("zero_masking",)
            for walkback in (1,)
            for walkback_jump in (False,)
            for autoencoding_loss in ("squared_error",)
            for contractive in (False,)
            for tied in (False,)
            for contractive_coef in ((0.001, 0.01, 0.1, 1) if contractive is True else (None,))
            for marginalized in (False,)
            for binarize_thresh in (None, 0.5)
        )
        all_params = list(all_params)
        print(len(all_params))
        nb = 0
        for p in all_params:
            p['model_name'] = 'model57'
            p['dataset'] = 'digits'
            p['budget_hours'] = 0.05
            cmd = build_cmd(launcher="scripts/launch_cpu", model_name="model57", dataset="digits", params=p, budget_hours=0.05)
            nb += job_write(p, cmd, where="test")
        return nb


def build_cmd(launcher="scripts/launch_gpu", model_name="model8", dataset="digits", params=None, prefix=None, budget_hours=None):
    summarized_name = summarize(params)
    if params is None:
        params = {}
    if prefix is None:
        prefix = "jobs/results/{}".format(summarized_name)
    name = "jobs/params/{}.json".format(summarized_name)

    output = "jobs/outputs/{}".format(summarized_name)
    extra = ""
    if budget_hours is not None:
        extra = "--budget-hours={}".format(budget_hours)
    cmd = "sbatch --time={} --output={} --error={} {} invoke train --update-db=1 --dataset={} --model-name={} --prefix={} --params={} {}"
    cmd = cmd.format(
        int(budget_hours * 60) + 15,
        output,
        output,
        launcher,
        dataset,
        model_name,
        prefix,
        summarized_name,
        extra
    )
    return cmd

if __name__ == "__main__":
    budget_hours = 10
    from lightjob.db import DB
    from lightjob.cli import get_dotfolder
    db = DB()
    folder = get_dotfolder()
    assert os.path.exists(folder)
    db.load(folder)

    def job_write(params, cmd, where=""):
        #db.job_update(summarize(params), {'type': 'training'})
        #return 0
        return db.safe_add_job(params, type='training', cmd=cmd, where=where)

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
                for nb_layers in (3,)
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
        nb = 0
        for p in all_params:
            p['model_name'] = 'model55'
            p['dataset'] = 'digits'
            p['budget_hours'] = budget_hours
            cmd = build_cmd(model_name="model55", dataset="digits", budget_hours=budget_hours, params=p)
            nb += job_write(p, cmd, where="jobset1")
        return nb

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
                for nb_layers in (2,)
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
        nb = 0
        for p in all_params:
            p['model_name'] = 'model56'
            p['dataset'] = 'digits'
            p['budget_hours'] = budget_hours
            cmd = build_cmd(model_name="model56", dataset="digits", params=p, budget_hours=budget_hours)
            nb += job_write(p, cmd, where="jobset2")
        return nb
    def jobset3():
        all_params = (
            build_params(
                OrderedDict(tied=tied,
                            use_wta_lifetime=use_wta_lifetime,
                            wta_lifetime_perc=wta_lifetime_perc,
                            nb_hidden_units=nb_hidden_units),
                denoise,
                noise,
                walkback,
                walkback_jump,
                autoencoding_loss,
                contractive,
                contractive_coef,
                marginalized,
                binarize_thresh)
            for nb_hidden_units in (1000,)
            for use_wta_lifetime in (True, False)
            for wta_lifetime_perc in ((0.02, 0.05, 0.1, 0.15) if use_wta_lifetime else (None,))
            for denoise in (None, 0.5)
            for noise in ("zero_masking",)
            for walkback in (1,)
            for walkback_jump in (False,)
            for autoencoding_loss in ("squared_error",)
            for contractive in ((True, False) if use_wta_lifetime is False else (False,))
            for tied in (True, False)
            for contractive_coef in ((0.001, 0.01, 0.1, 1) if contractive is True else (None,))
            for marginalized in (False,)
            for binarize_thresh in (None, 0.5)
        )
        all_params = list(all_params)
        print(len(all_params))
        nb = 0
        for p in all_params:
            p['model_name'] = 'model57'
            p['dataset'] = 'digits'
            p['budget_hours'] = budget_hours
            cmd = build_cmd(model_name="model57", dataset="digits", params=p, budget_hours=budget_hours)
            nb += job_write(p, cmd, where="jobset3")
        return nb

    def jobset4():
        all_params = (
            build_params(
                OrderedDict(tied=tied,
                            use_wta_lifetime=use_wta_lifetime,
                            wta_lifetime_perc=wta_lifetime_perc,
                            nb_hidden_units=nb_hidden_units),
                denoise,
                noise,
                walkback,
                walkback_jump,
                autoencoding_loss,
                contractive,
                contractive_coef,
                marginalized,
                binarize_thresh)
            for nb_hidden_units in (1000,)
            for use_wta_lifetime in (False,)
            for wta_lifetime_perc in (None,)
            for denoise in (None, 0.5)
            for noise in ("zero_masking",)
            for walkback in (1,)
            for walkback_jump in (False,)
            for autoencoding_loss in ("squared_error",)
            for contractive in (True,)
            for tied in (True,)
            for contractive_coef in ((0.2, 0.5, 0.7, 0.9, 2, 3, 4, 5, 10, 20, 30, 50, 80) if contractive is True else (None,))
            for marginalized in (False,)
            for binarize_thresh in (None,)
        )
        all_params = list(all_params)
        print(len(all_params))
        nb = 0
        for p in all_params:
            p['model_name'] = 'model57'
            p['dataset'] = 'digits'
            p['budget_hours'] = budget_hours
            cmd = build_cmd(model_name="model57", dataset="digits", params=p, budget_hours=budget_hours)
            nb += job_write(p, cmd, where="jobset4")
            print(p)
        return nb

    def jobset5():
        import numpy as np
        C = np.linspace(0, 1, 50).tolist()
        C.extend(np.linspace(1, 2, 50).tolist())
        C.extend(np.linspace(2, 5, 50).tolist())

        C = sorted(C)
        print(C)
        all_params = (
            build_params(
                OrderedDict(tied=tied,
                            use_wta_lifetime=use_wta_lifetime,
                            wta_lifetime_perc=wta_lifetime_perc,
                            nb_hidden_units=nb_hidden_units),
                denoise,
                noise,
                walkback,
                walkback_jump,
                autoencoding_loss,
                contractive,
                contractive_coef,
                marginalized,
                binarize_thresh)
            for nb_hidden_units in (1000,)
            for use_wta_lifetime in (False,)
            for wta_lifetime_perc in (None,)
            for denoise in (None,)
            for noise in ("zero_masking",)
            for walkback in (1,)
            for walkback_jump in (False,)
            for autoencoding_loss in ("squared_error",)
            for contractive in (True,)
            for tied in (False,)
            for contractive_coef in (C if contractive is True else (None,))
            for marginalized in (False,)
            for binarize_thresh in (None,)
        )
        all_params = list(all_params)
        print(len(all_params))
        nb = 0
        for p in all_params:
            p['model_name'] = 'model57'
            p['dataset'] = 'digits'
            p['budget_hours'] = budget_hours
            cmd = build_cmd(model_name="model57", dataset="digits", params=p, budget_hours=budget_hours)
            nb += job_write(p, cmd, where="jobset5")
            print(p)
        return nb

    nb = 0
    #nb += test()
    #nb += jobset1()
    #nb += jobset2()
    #nb += jobset3()
    #nb += jobset4()
    nb += jobset5()
    print("Total number of jobs added : {}".format(nb))
