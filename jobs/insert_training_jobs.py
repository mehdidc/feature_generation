import json
from collections import OrderedDict
from lightjob.utils import summarize
from lightjob.db import SUCCESS
from hp import get_next_skopt, Categorical

import numpy as np
import random


import click

from lightjob.cli import load_db

db = load_db()
budget_hours = 10  # default budget hours


def build_cmd(launcher="scripts/launch_gpu",
              model_name="model8",
              dataset="digits",
              params=None,
              prefix=None,
              budget_hours=None,
              force_w=None,
              force_h=None):
    summarized_name = summarize(params)
    if params is None:
        params = {}
    if prefix is None:
        prefix = "jobs/results/{}".format(summarized_name)

    output = "jobs/outputs/{}".format(summarized_name)
    extra = []
    if budget_hours is not None:
        extra.append("--budget-hours={} ".format(budget_hours))
    if force_w is not None:
        extra.append("--force-w={}".format(force_w))
    if force_h is not None:
        extra.append("--force-h={}".format(force_h))
    extra = " ".join(extra)
    cmd = "sbatch --time={} --output={} --error={} {} invoke train --update-db=1 --dataset={} --model-name={} --prefix={} --params={} {}"
    cmd = cmd.format(
        int(budget_hours * 60) + 60,
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


def job_write(params, cmd, where="", dry=False):
    if dry:
        return 0
    print(json.dumps(params, indent=4))
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
    """
    Exploring params of conv autoenc
    """
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
        cmd = build_cmd(model_name="model55", dataset="digits",
                        budget_hours=budget_hours, params=p)
        nb += job_write(p, cmd, where="jobset1")
    return nb


def jobset2():
    """
    Exploring various params of fully connected autoenc
    """
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
        cmd = build_cmd(model_name="model56", dataset="digits",
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where="jobset2")
    return nb


def jobset3():
    """
    Exploring params of fully connected autoenc
    """
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
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where="jobset3")
    return nb


def jobset4():
    """
    Exploring params of contraction coef with tiying fully connected autoenc
    """
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
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where="jobset4")
        print(p)
    return nb


def jobset5():
    """
    Exploring contraction coef heavily without tiying fully connected autoenc
    """

    import numpy as np
    C = np.linspace(0, 1, 50).tolist()
    C.extend(np.linspace(1, 2, 50).tolist())
    C.extend(np.linspace(2, 5, 50).tolist())
    C.extend(np.linspace(5, 10, 30).tolist())
    C.extend(np.linspace(10, 100, 50).tolist())

    C = sorted(C)

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
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where="jobset5")
        print(p)
    return nb


def jobset6():
    """
    Exploring params of denoising with walkback
    """
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
        for denoise in (0.1, 0.2, 0.3, 0.4, 0.5)
        for noise in ("zero_masking",)
        for walkback in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        for walkback_jump in (True, False)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for tied in (False,)
        for contractive_coef in (None,)
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
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where="jobset6")
        print(p)
    return nb


def jobset7():
    """
    Exploring params of sparsity without contraction
    """

    import numpy as np
    C = np.linspace(0, 1, 100)
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
        for use_wta_lifetime in (True,)
        for wta_lifetime_perc in C
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for tied in (False,)
        for contractive_coef in (None,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = 3
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=3)
        nb += job_write(p, cmd, where="jobset7")
        print(p)
    return nb


def jobset8():
    """
    Exploring params of sparsity without contraction (this was en error so it is a duplicate of jobset7 i forgot to make contractive to True)
    so forget about this jobset
    """
    import numpy as np
    C = np.linspace(0, 1, 100)
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
        for use_wta_lifetime in (True,)
        for wta_lifetime_perc in C
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for tied in (False,)
        for contractive_coef in (1,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = 48
        p['optimization'] = dict(max_nb_epochs=72180)
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=6)
        nb += job_write(p, cmd, where="jobset8")
        print(p)
    return nb


def jobset9():
    """
    Exploring params of denoising without contraction
    """
    import numpy as np
    C = np.linspace(0, 1, 30)
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
        for denoise in C
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for tied in (False,)
        for contractive_coef in (1,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = 3
        p['optimization'] = dict(max_nb_epochs=72180)
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=p['budget_hours'])
        nb += job_write(p, cmd, where="jobset9")
        print(p)
    return nb


def jobset10():
    """
    Exploring params of denoising with contraction=1
    """
    import numpy as np
    C = np.linspace(0, 1, 30)
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
        for denoise in C
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (True,)
        for tied in (False,)
        for contractive_coef in (1,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = 7
        p['optimization'] = dict(max_nb_epochs=72180)
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=p['budget_hours'])
        nb += job_write(p, cmd, where="jobset10")
        print(p)
    return nb


def jobset11():
    """
    Exploring params of sparsity with contraction=1
    """
    import numpy as np
    C = np.linspace(0, 1, 30)
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
        for use_wta_lifetime in (True,)
        for wta_lifetime_perc in C
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (True,)
        for tied in (False,)
        for contractive_coef in (1,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = 7
        p['optimization'] = dict(max_nb_epochs=72180)
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=p['budget_hours'])
        nb += job_write(p, cmd, where="jobset11")
        print(p)
    return nb


def jobset12():
    """
    Exploring params of sparsity with contraction=35.714285714285715
    """
    import numpy as np
    C = np.linspace(0, 1, 30)
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
        for use_wta_lifetime in (True,)
        for wta_lifetime_perc in C
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (True,)
        for tied in (False,)
        for contractive_coef in (35.714285714285715,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = 7
        p['optimization'] = dict(max_nb_epochs=72180)
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=p['budget_hours'])
        nb += job_write(p, cmd, where="jobset12")
        print(p)
    return nb


def jobset13():
    """
    Exploring params of denoising with contraction=35.714285714285715
    """
    import numpy as np
    C = np.linspace(0, 1, 30)
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
        for denoise in C
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (True,)
        for tied in (False,)
        for contractive_coef in (35.714285714285715,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = 7
        p['optimization'] = dict(max_nb_epochs=72180)
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=p['budget_hours'])
        nb += job_write(p, cmd, where="jobset13")
        print(p)
    return nb


def jobset14():
    """
    Exploring params of conv autoenc v2
    """
    nb_filters_per_layer = {
        1: [64],
        2: [64, 64],
        3: [64, 64, 64],
        4: [64, 64, 64, 64],
        5: [64, 64, 64, 64, 64],
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
        for nb_layers in (1, 2, 3, 4, 5)
        for filter_size in (5,)
        for use_wta_spatial in (True,)
        for use_wta_channel in (True, False)
        if use_wta_spatial is True or use_wta_channel is True
        for nb_filters_mul in (1,)
        for wta_channel_stride in ((1, 2, 3, 4) if use_wta_channel else (1,))
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in ((1, 3, 5) if denoise is not None else (1,))
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for contractive_coef in (0,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    nb = 0
    budget_hours = 6
    for p in all_params:
        p['model_name'] = 'model59'
        p['dataset'] = 'digits'
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name="model59", dataset="digits",
                        budget_hours=budget_hours, params=p)
        nb += job_write(p, cmd, where="jobset14")
    return nb


def jobset15():
    """
    Exploring params of sparsity with contraction=35.714285714285715 without limitation in nb of epochs
    """
    import numpy as np
    C = np.linspace(0, 1, 30)
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
        for use_wta_lifetime in (True,)
        for wta_lifetime_perc in C
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (True,)
        for tied in (False,)
        for contractive_coef in (35.714285714285715,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = 10
        #p['optimization'] = dict(max_nb_epochs=72180)
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=p['budget_hours'])
        nb += job_write(p, cmd, where="jobset15")
        print(p)
    return nb


def jobset16():
    """
    Exploring params of denoising with contraction=35.714285714285715 without limitation in nb epochs
    """
    import numpy as np
    C = np.linspace(0, 1, 30)
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
        for denoise in C
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_jump in (False,)
        for autoencoding_loss in ("squared_error",)
        for contractive in (True,)
        for tied in (False,)
        for contractive_coef in (35.714285714285715,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = 10
        #p['optimization'] = dict(max_nb_epochs=72180)
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=p['budget_hours'])
        nb += job_write(p, cmd, where="jobset16")
        print(p)
    return nb


def jobset17():
    """
    Exploring params of denoising with walkback (like jobset6 but with the correct walkback used in bengio)
    """
    all_params = (
        OrderedDict(
            model_params=OrderedDict(tied=tied,
                                     use_wta_lifetime=use_wta_lifetime,
                                     wta_lifetime_perc=wta_lifetime_perc,
                                     nb_hidden_units=nb_hidden_units),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_hidden_units in (1000,)
        for use_wta_lifetime in (False,)
        for wta_lifetime_perc in (None,)
        for denoise in (0.1, 0.2, 0.3, 0.4, 0.5)
        for noise in ("salt_and_pepper",)
        for walkback in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for tied in (False,)
        for contractive_coef in (None,)
        for marginalized in (False,)
        for binarize_thresh in (0.5,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 6
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'digits'
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name="model57", dataset="digits",
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where="jobset17")
        print(json.dumps(p, indent=4))
    return nb


def jobset19():
    """
    Exploring params of denoising fixed to 0.5 but  with varying hidden layers and nb of hidden units
    """
    all_params = (
        OrderedDict(
            model_params=OrderedDict(
                use_wta_lifetime=use_wta_lifetime,
                wta_lifetime_perc=wta_lifetime_perc,
                nb_layers=nb_layers,
                nb_hidden_units=nb_hidden_units),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_hidden_units in (100, 500, 600, 700, 800, 1000, 1200, 1600, 2000, 3000, 4000)
        for nb_layers in (1, 2, 3, 4)
        for use_wta_lifetime in (False,)
        for wta_lifetime_perc in (None,)
        for denoise in (0.5,)
        for noise in ("salt_and_pepper",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for contractive_coef in (None,)
        for marginalized in (False,)
        for binarize_thresh in (0.5,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 6
    for p in all_params:
        p['model_name'] = 'model56'
        p['dataset'] = 'digits'
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name="model56", dataset="digits",
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where="jobset19")
        print(json.dumps(p, indent=4))
    return nb


def jobset20():
    """
    Exploring params of denoising and nb hidden units with contraction=1
    """
    import numpy as np
    C = np.linspace(0, 1, 10)
    all_params = (
        OrderedDict(
            model_params=OrderedDict(tied=tied,
                                     use_wta_lifetime=use_wta_lifetime,
                                     wta_lifetime_perc=wta_lifetime_perc,
                                     nb_hidden_units=nb_hidden_units),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_hidden_units in (500, 1000, 3000, 4000)
        for use_wta_lifetime in (False,)
        for wta_lifetime_perc in (None,)
        for denoise in C
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (True,)
        for tied in (False,)
        for contractive_coef in (1.,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 6
    model_name = 'model57'
    dataset = 'digits'
    jobset_name = "jobset21"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb


def jobset21():
    """
    Exploring nb of filters around the best conv archi found in jobset1 (checkeed visually)
    """
    nb_filters_per_layer = {
        3: [[2**x, 2**(x + 1), 2**(x + 2)] for x in range(3, 9)],
        4: [[2**x, 2**(x + 1), 2**(x + 1), 2**(x + 2)] for x in range(3, 9)],
        5: [[2**x, 2**x, 2**(x + 1), 2**(x + 1), 2**(x + 2)] for x in range(3, 9)]
    }
    all_params = (
        OrderedDict(
            model_params=OrderedDict(nb_filters=nb_filters,
                                     filter_size=filter_size,
                                     use_wta_channel=use_wta_channel,
                                     use_wta_spatial=use_wta_spatial,
                                     nb_filters_mul=nb_filters_mul,
                                     wta_channel_stride=wta_channel_stride,
                                     nb_layers=nb_layers),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_filters_mul in (1,)
        for filter_size in (5,)
        for nb_layers in (3, 4, 5)
        for nb_filters in nb_filters_per_layer[nb_layers]
        for use_wta_channel in (True,)
        for use_wta_spatial in (False,)
        for wta_channel_stride in (2, 4)
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for contractive_coef in (None,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 6
    model_name = 'model55'
    dataset = 'digits'
    jobset_name = "jobset21"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb


def jobset22():
    """
    like jobset20 but with limitation in nb of epochs to obtain exact same result of the 'good' contraction coef case
    """
    import numpy as np
    C = np.linspace(0, 1, 10)
    all_params = (
        OrderedDict(
            model_params=OrderedDict(tied=tied,
                                     use_wta_lifetime=use_wta_lifetime,
                                     wta_lifetime_perc=wta_lifetime_perc,
                                     nb_hidden_units=nb_hidden_units),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_hidden_units in (500, 1000, 3000, 4000)
        for use_wta_lifetime in (False,)
        for wta_lifetime_perc in (None,)
        for denoise in C
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (True,)
        for tied in (False,)
        for contractive_coef in (1.,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 7
    model_name = 'model57'
    dataset = 'digits'
    jobset_name = "jobset22"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        p['optimization'] = dict(max_nb_epochs=72180)
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb


def jobset23():
    """
    Exactly jobset5 but for fonts
    """
    import numpy as np
    C = np.linspace(0, 10, 30).tolist()
    C.extend(np.linspace(10, 100, 10).tolist())

    C = sorted(C)

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
    budget_hours = 6
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'fonts'
        p['force_w'] = 28
        p['force_h'] = 28
        p['mode'] = 'minibatch'
        p['budget_hours'] = budget_hours
        p['optimization'] = dict(max_nb_epochs=1000)
        cmd = build_cmd(model_name="model57", dataset="fonts", params=p,
                        budget_hours=budget_hours, force_w=28, force_h=28)
        nb += job_write(p, cmd, where="jobset23")
        print(p)
    return nb


def jobset24():
    """
    Exploring nb of filters around the best conv archi found in jobset21 : id of the best conv archi starts by '7fce'
    """
    from itertools import product
    all_nb_filters = list(product([32, 64, 128], repeat=3))
    all_params = (
        OrderedDict(
            model_params=OrderedDict(nb_filters=nb_filters,
                                     filter_size=filter_size,
                                     use_wta_channel=use_wta_channel,
                                     use_wta_spatial=use_wta_spatial,
                                     nb_filters_mul=nb_filters_mul,
                                     wta_channel_stride=wta_channel_stride,
                                     nb_layers=nb_layers),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_filters_mul in (1,)
        for filter_size in (5,)
        for nb_layers in (3,)
        for nb_filters in all_nb_filters
        for use_wta_channel in (True, False)
        for use_wta_spatial in (False,)
        for wta_channel_stride in (4,)
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for contractive_coef in (None,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 6
    model_name = 'model55'
    dataset = 'digits'
    jobset_name = "jobset24"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb


def jobset25():
    """
    exactly jobset21 but for chinese_icdar
    """
    nb_filters_per_layer = {
        3: [[2**x, 2**(x + 1), 2**(x + 2)] for x in range(3, 9)],
        4: [[2**x, 2**(x + 1), 2**(x + 1), 2**(x + 2)] for x in range(3, 9)],
        5: [[2**x, 2**x, 2**(x + 1), 2**(x + 1), 2**(x + 2)] for x in range(3, 9)]
    }
    all_params = (
        OrderedDict(
            model_params=OrderedDict(nb_filters=nb_filters,
                                     filter_size=filter_size,
                                     use_wta_channel=use_wta_channel,
                                     use_wta_spatial=use_wta_spatial,
                                     nb_filters_mul=nb_filters_mul,
                                     wta_channel_stride=wta_channel_stride,
                                     nb_layers=nb_layers),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_filters_mul in (1,)
        for filter_size in (5,)
        for nb_layers in (3, 4, 5)
        for nb_filters in nb_filters_per_layer[nb_layers]
        for use_wta_channel in (True,)
        for use_wta_spatial in (False,)
        for wta_channel_stride in (2, 4)
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for contractive_coef in (None,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 6
    model_name = 'model55'
    dataset = 'chinese_icdar'
    jobset_name = "jobset25"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        p['force_w'] = 28
        p['force_h'] = 28
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb


def jobset26():
    """
    Exactly jobset5 but for chinese
    """
    import numpy as np
    C = np.linspace(0, 10, 30).tolist()
    C.extend(np.linspace(10, 100, 10).tolist())

    C = sorted(C)

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
    budget_hours = 6
    for p in all_params:
        p['model_name'] = 'model57'
        p['dataset'] = 'chinese_icdar'
        p['force_w'] = 28
        p['force_h'] = 28
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name="model57", dataset="chinese_icdar",
                        params=p, budget_hours=budget_hours, force_w=28, force_h=28)
        nb += job_write(p, cmd, where="jobset26")
        print(p)
    return nb


def jobset27():
    """
    same than jobset20 but with sparsity
    """
    import numpy as np
    C = np.linspace(0.5, 1, 6)
    all_params = (
        OrderedDict(
            model_params=OrderedDict(tied=tied,
                                     use_wta_lifetime=use_wta_lifetime,
                                     wta_lifetime_perc=wta_lifetime_perc,
                                     nb_hidden_units=nb_hidden_units),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_hidden_units in (500, 1000, 3000, 4000)
        for use_wta_lifetime in (True,)
        for wta_lifetime_perc in C
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (True, False)
        for tied in (False,)
        for contractive_coef in (1.,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 6
    model_name = 'model57'
    dataset = 'digits'
    jobset_name = "jobset27"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb


def jobset28():
    """
    same than jobset27 but different range of params and wihtout contraction
    """
    import numpy as np
    C = np.linspace(0, 0.3, 10)
    all_params = (
        OrderedDict(
            model_params=OrderedDict(tied=tied,
                                     use_wta_lifetime=use_wta_lifetime,
                                     wta_lifetime_perc=wta_lifetime_perc,
                                     nb_hidden_units=nb_hidden_units),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_hidden_units in (500, 1000, 3000, 4000)
        for use_wta_lifetime in (True,)
        for wta_lifetime_perc in C
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for tied in (False,)
        for contractive_coef in (1.,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 10
    model_name = 'model57'
    dataset = 'digits'
    jobset_name = "jobset28"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb


def jobset29():
    """
    same than jobset28 but with another optimization algo (adam, was verified experimentally that it leads to better features)
    """
    import numpy as np
    C = np.linspace(0, 0.3, 10)
    all_params = (
        OrderedDict(
            model_params=OrderedDict(tied=tied,
                                     use_wta_lifetime=use_wta_lifetime,
                                     wta_lifetime_perc=wta_lifetime_perc,
                                     nb_hidden_units=nb_hidden_units),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_hidden_units in (1000, 3000, 4000)
        for use_wta_lifetime in (True,)
        for wta_lifetime_perc in C
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for tied in (False,)
        for contractive_coef in (1.,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 2
    model_name = 'model57'
    dataset = 'digits'
    jobset_name = "jobset29"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        p['optimization'] = dict(algo='adam', initial_lr=0.01)
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb


def jobset30():
    """
    Exploring params of denoising fixed to 0.5 but  with varying hidden layers and nb of hidden units
    """
    all_params = (
        OrderedDict(
            model_params=OrderedDict(
                use_wta_lifetime=use_wta_lifetime,
                wta_lifetime_perc=wta_lifetime_perc,
                nb_layers=nb_layers,
                out_nonlin='tanh',
                nb_hidden_units=nb_hidden_units),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_hidden_units in (100, 500, 600, 700, 800, 1000, 1200, 1600, 2000, 3000, 4000)
        for nb_layers in (1, 2, 3, 4)
        for use_wta_lifetime in (False,)
        for wta_lifetime_perc in (None,)
        for denoise in (0.5,)
        for noise in ("superpose",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for contractive_coef in (None,)
        for marginalized in (False,)
        for binarize_thresh in (0.5,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 6
    for p in all_params:
        p['model_name'] = 'model56'
        p['dataset'] = 'digits'
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name="model56", dataset="digits",
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where="jobset30")
        print(json.dumps(p, indent=4))
    return nb


def jobset31():
    """
    """
    import numpy as np
    C = np.linspace(0, 1, 20)
    all_params = (
        OrderedDict(
            model_params=OrderedDict(
                use_wta_sparse=True,
                wta_sparse_perc=wta_sparse_perc,
                nb_layers=nb_layers,
                nb_hidden_units=nb_hidden_units),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_hidden_units in (1000, 4000)
        for nb_layers in (1, 2, 3)
        for wta_sparse_perc in C
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for tied in (False,)
        for contractive_coef in (1.,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 2
    model_name = 'model64'
    dataset = 'digits'
    jobset_name = "jobset31"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        p['optimization'] = dict(algo='adam', initial_lr=0.001)
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb


def jobset32():
    """
    jobset1 for fonts
    """
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
    budget_hours = 10
    for p in all_params:
        p['model_name'] = 'model55'
        p['budget_hours'] = budget_hours
        p['dataset'] = 'fonts'
        p['force_w'] = 28
        p['force_h'] = 28
        p['mode'] = 'minibatch'
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name="model55", dataset="fonts", params=p,
                        budget_hours=budget_hours, force_w=28, force_h=28)
        nb += job_write(p, cmd, where="jobset32")
        print(p)

    return nb


def jobset33():
    """
    jobset21 but with more training time
    """
    nb_filters_per_layer = {
        3: [[2**x, 2**(x + 1), 2**(x + 2)] for x in range(3, 9)],
        4: [[2**x, 2**(x + 1), 2**(x + 1), 2**(x + 2)] for x in range(3, 9)],
        5: [[2**x, 2**x, 2**(x + 1), 2**(x + 1), 2**(x + 2)] for x in range(3, 9)]
    }
    all_params = (
        OrderedDict(
            model_params=OrderedDict(nb_filters=nb_filters,
                                     filter_size=filter_size,
                                     use_wta_channel=use_wta_channel,
                                     use_wta_spatial=use_wta_spatial,
                                     nb_filters_mul=nb_filters_mul,
                                     wta_channel_stride=wta_channel_stride,
                                     nb_layers=nb_layers),
            denoise=denoise,
            noise=noise,
            walkback=walkback,
            walkback_mode=walkback_mode,
            autoencoding_loss=autoencoding_loss,
            contractive=contractive,
            contractive_coef=contractive_coef,
            marginalized=marginalized,
            binarize_thresh=binarize_thresh)
        for nb_filters_mul in (1,)
        for filter_size in (5,)
        for nb_layers in (3, 4, 5)
        for nb_filters in nb_filters_per_layer[nb_layers]
        for use_wta_channel in (True,)
        for use_wta_spatial in (False,)
        for wta_channel_stride in (2, 4)
        for denoise in (None,)
        for noise in ("zero_masking",)
        for walkback in (1,)
        for walkback_mode in ('bengio_without_sampling',)
        for autoencoding_loss in ("squared_error",)
        for contractive in (False,)
        for contractive_coef in (None,)
        for marginalized in (False,)
        for binarize_thresh in (None,)
    )
    all_params = list(all_params)
    print(len(all_params))
    nb = 0
    budget_hours = 12
    model_name = 'model55'
    dataset = 'digits'
    jobset_name = "jobset33"
    for p in all_params:
        p['model_name'] = model_name
        p['dataset'] = dataset
        p['budget_hours'] = budget_hours
        cmd = build_cmd(model_name=model_name, dataset=dataset,
                        params=p, budget_hours=budget_hours)
        nb += job_write(p, cmd, where=jobset_name)
        print(json.dumps(p, indent=4))
    return nb

# hyperopt loop for model64

def jobset34():
    model = 'model64'
    crit = 'knn_classification_accuracy'

    def get_crit(j):
        # minus because we want to maximize
        return -float(np.min(j[crit].values()))

    def filter_func(job):
        if job['state'] != SUCCESS:
            return False
        if 'stats' not in job:
            return False
        if crit not in job['stats']:
            return False
        if job['content']['model_name'] != model:
            return False
        return True
    jobs = db.jobs_filter(filter_func, where='jobset34')
    inputs = [j['content'] for j in jobs]
    outputs = [get_crit(j['stats']) for j in jobs]
    print('len of prior params : {}'.format(len(inputs)))

    space_skopt = [
        Categorical([True, False]),  # use_wta_sparse
        [0., 1.],  # use_wta_sparse_perc
        [1, 5],  # nb_layers
        [100, 2100],  # nb_hidden_units
        [0., 1.],  # denoise
        Categorical([True, False]),  # denoise_activated
        ['zero_masking', 'salt_and_pepper'],
        [1, 5],  # walkback
        Categorical([None, 0.5])  # binarize_thresh
    ]

    def to_skopt(params):
        return [
            params['model_params']['use_wta_sparse'],
            params['model_params']['wta_sparse_perc'],
            params['model_params']['nb_layers'],
            params['model_params']['nb_hidden_units'],
            params['denoise'] if params['denoise'] is not None else 0,
            params['denoise'] is not None, #whether denoise is activated or not
            params['noise'],
            params['walkback'],
            params['binarize_thresh']
        ]

    def from_skopt(params):

        model_params = OrderedDict(
            use_wta_sparse=params[0],
            wta_sparse_perc=params[1],
            nb_layers=params[2],
            nb_hidden_units=params[3]
        )
        params = OrderedDict(
            model_params=model_params,
            denoise=(params[4] if params[5] else None),
            noise=params[6],
            walkback=params[7],
            walkback_mode='bengio_without_sampling',
            autoencoding_loss='squared_error',
            mode='minibatch',
            contractive=False,
            contractive_coef=None,
            marginalized=False,
            binarize_thresh=params[8],
            model_name='model64',
            dataset='digits',
            budget_hours=4,
        )
        return params

    inputs_ = map(to_skopt, inputs)
    seed = np.random.randint(0, 99999999)
    rng = np.random.RandomState(seed)
    hp_next = get_next_skopt(
        inputs_,
        outputs,
        space_skopt,
        rstate=rng)
    params_next = from_skopt(hp_next)
    budget_hours = params_next['budget_hours']
    model_name = params_next['model_name']
    dataset = params_next['dataset']
    jobset_name = "jobset34"
    cmd = build_cmd(model_name=model_name,
                    dataset=dataset,
                    params=params_next,
                    budget_hours=budget_hours)
    nb = job_write(params_next, cmd, where=jobset_name)
    return nb


def jobset35():
    # vertebrate convnet hyperopt search
    rng = random
    nb_layers = rng.randint(1, 7)
    nb_filters = [2 ** rng.randint(5, 9) for _ in range(nb_layers)]
    model_params = OrderedDict(
        nb_layers=nb_layers,
        nb_filters=nb_filters,
        filter_size=rng.choice((3, 5)),
        use_channel=rng.choice((True, False)),
        use_spatial=True,
        spatial_k=rng.randint(1, 10),
        channel_stride=rng.choice((1, 2, 4)),
        weight_sharing=rng.choice((True, False)),
        merge_op=rng.choice(('sum', 'mul'))
    )
    params = OrderedDict(
        model_params=model_params,
        denoise=None,
        noise=None,
        walkback=1,
        walkback_mode='bengio_without_sampling',
        autoencoding_loss='squared_error',
        mode='random',
        contractive=False,
        contractive_coef=None,
        marginalized=False,
        binarize_thresh=None
    )
    budget_hours = 4
    model_name = 'model73'
    dataset = 'digits'
    jobset_name = "jobset35"

    params['model_name'] = model_name
    params['dataset'] = 'digits'
    params['budget_hours'] = budget_hours

    cmd = build_cmd(model_name=model_name,
                    dataset=dataset,
                    params=params,
                    budget_hours=budget_hours)
    nb = job_write(params, cmd, where=jobset_name)
    return nb


def jobset36():
    # Continuous brush stroke hyper-search
    rng = random
    nb_layers = rng.randint(1, 7)
    nb_units = [rng.randint(1, 20) * 100 for l in range(nb_layers)]
    model_params = OrderedDict(
        nb_layers=nb_layers,
        nb_units=nb_units,
        n_steps=rng.randint(1, 30),
        patch_size=rng.randint(1, 9),
        nonlin=rng.choice(('rectify', 'very_leaky_rectify'))
    )
    params = OrderedDict(
        model_params=model_params,
        denoise=None,
        noise=None,
        walkback=1,
        walkback_mode='bengio_without_sampling',
        autoencoding_loss='squared_error',
        mode='random',
        contractive=False,
        contractive_coef=None,
        marginalized=False,
        binarize_thresh=None
    )
    budget_hours = 4
    model_name = 'model75'
    dataset = 'digits'
    jobset_name = "jobset36"

    params['model_name'] = model_name
    params['dataset'] = 'digits'
    params['budget_hours'] = budget_hours

    cmd = build_cmd(model_name=model_name,
                    dataset=dataset,
                    params=params,
                    budget_hours=budget_hours)
    nb = job_write(params, cmd, where=jobset_name)
    return b


def jobset37():
    import math
    # discrete brush stroke hyper-search

    values = db.get_values('stats.training.test_recons_error', where='jobset37')
    values = list(values)
    inputs = [v['job']['content'] for v in values]
    outputs = [v['stats.training.test_recons_error'] for v in values]

    space_skopt = [
        (1, 7),  # nb_layers
        (5, 9),  # nb_filters1 expo
        (5, 9),  # nb_filters2 expo
        (5, 9),  # nb_filters3 expo
        (5, 9),  # nb_filters4 expo
        (5, 9),  # nb_filters5 expo
        (5, 9),  # nb_filters6 expo
        (5, 9),  # nb_filters7 expo
        Categorical([3, 5]),  # filter_size
        Categorical(['rectify', 'very_leaky_rectify']),
    ]

    def from_skopt(params):
        nb_layers = params[0]
        nb_filters = params[1:8]
        nb_filters = map(lambda x: 2**x, nb_filters)
        nb_filters = nb_filters[0:nb_layers]
        filter_size = params[8]
        nonlin = params[9]
        model_params = OrderedDict(
            nb_layers=nb_layers,
            nb_filters=nb_filters,
            filter_size=filter_size,
            nonlin=nonlin
        )
        params = OrderedDict(
            model_params=model_params,
            denoise=None,
            noise=None,
            walkback=1,
            walkback_mode='bengio_without_sampling',
            autoencoding_loss='squared_error',
            mode='random',
            contractive=False,
            contractive_coef=None,
            marginalized=False,
            binarize_thresh=None,
            budget_hours=8,
            model_name='model76',
            dataset='digits',
        )
        return params

    def to_skopt(params):
        m = params['model_params']
        nb_filters = m['nb_filters'] + [5] * (7 - len(m['nb_filters']))
        nb_filters = map(lambda x:int(math.log(x, 2)), nb_filters)
        p = [m['nb_layers']] + nb_filters + [m['filter_size'], m['nonlin']]
        return p

    inputs_ = map(to_skopt, inputs)
    seed = np.random.randint(0, 99999999)
    rng = np.random.RandomState(seed)
    hp_next = get_next_skopt(
        inputs_,
        outputs,
        space_skopt,
        rstate=rng)
    params_next = from_skopt(hp_next)
    budget_hours = params_next['budget_hours']
    model_name = params_next['model_name']
    dataset = params_next['dataset']
    jobset_name = "jobset37"
    cmd = build_cmd(model_name=model_name,
                    dataset=dataset,
                    params=params_next,
                    budget_hours=budget_hours)
    nb = job_write(params_next, cmd, where=jobset_name)
    return nb


def jobset_recurrent_brush_stroke(jobset_name, model_name, update=lambda p:p):
    # Continuous brush stroke with recurrent layers hyper-search
    rng = random
    nb_fc_layers = rng.randint(1, 4)
    nb_fc_units = [rng.randint(1, 20) * 100 for l in range(nb_fc_layers)]
    nb_recurrent_layers = rng.randint(1, 3)
    nb_recurrent_units = [rng.randint(1, 5) * 100 for l in range(nb_recurrent_layers)]

    model_params = OrderedDict(
        nb_fc_layers=nb_fc_layers,
        nb_fc_units=nb_fc_units,
        nb_recurrent_layers=nb_recurrent_layers,
        nb_recurrent_units=nb_recurrent_units,
        n_steps=rng.randint(1, 64),
        patch_size=rng.randint(1, 6),
        nonlin=rng.choice(('rectify', 'very_leaky_rectify'))
    )
    params = OrderedDict(
        model_params=model_params,
        denoise=None,
        noise=None,
        walkback=1,
        walkback_mode='bengio_without_sampling',
        autoencoding_loss='squared_error',
        mode='random',
        contractive=False,
        contractive_coef=None,
        marginalized=False,
        binarize_thresh=None,
        optimization=dict(max_nb_epochs=9999999999)
    )
    budget_hours = 10
    model_name = model_name
    dataset = 'digits'
    jobset_name = jobset_name

    params['model_name'] = model_name
    params['dataset'] = dataset
    params['budget_hours'] = budget_hours

    params = update(params)
    dataset = params['dataset']

    cmd = build_cmd(model_name=model_name,
                    dataset=dataset,
                    params=params,
                    budget_hours=budget_hours)
    nb = job_write(params, cmd, where=jobset_name)
    return nb


def jobset38():
    return jobset_recurrent_brush_stroke('jobset38', 'model77')


def jobset39():
    return jobset_recurrent_brush_stroke('jobset39', 'model78')


def jobset40():

    def update(params):
        params['model_params']['stride'] = False
        return params
    return jobset_recurrent_brush_stroke('jobset40', 'model81', update=update)


def jobset41():

    def update(params):
        rng = random
        params['model_params']['stride'] = False
        params['model_params']['sigma'] = rng.choice((None, 1))
        params['model_params']['normalize'] = rng.choice(('sigmoid', 'maxmin'))
        params['model_params']['reduce'] = rng.choice(('sum', 'over'))
        return params
    return jobset_recurrent_brush_stroke('jobset41', 'model81', update=update)


def jobset42():

    def update(params):
        rng = random
        params['model_params']['stride'] = False
        params['model_params']['sigma'] = 1
        params['model_params']['normalize'] = 'sigmoid'
        params['model_params']['reduce'] = 'correct_over'
        params['model_params']['alpha'] = rng.uniform(0, 1)
        return params
    return jobset_recurrent_brush_stroke('jobset42', 'model81', update=update)


def jobset43():

    def update(params):
        rng = random
        params['model_params']['stride'] = False
        params['model_params']['sigma'] = rng.choice((None, 1))
        params['model_params']['normalize'] = 'sigmoid'
        params['model_params']['reduce'] = 'max'
        return params
    return jobset_recurrent_brush_stroke('jobset43', 'model81', update=update)


def jobset44():

    def update(params):
        rng = random
        params['model_params']['stride'] = False
        params['model_params']['sigma'] = rng.choice((None, 1))
        params['model_params']['normalize'] = 'sigmoid'
        params['model_params']['out_reduce'] = rng.choice(('sum', 'over', 'max'))
        params['model_params']['inp_reduce'] = rng.choice(('sum', 'over', 'prev'))
        del params['model_params']['nb_recurrent_layers']
        params['model_params']['nb_recurrent_units'] = params['model_params']['nb_recurrent_units'][0]
        return params
    return jobset_recurrent_brush_stroke('jobset44', 'model82', update=update)


def jobset45():

    def update(params):
        rng = random
        params['model_params']['stride'] = False
        params['model_params']['sigma'] = rng.choice((None, 1))
        params['model_params']['normalize'] = 'sigmoid'
        params['model_params']['reduce'] = rng.choice(('sum', 'over', 'max'))
        params['model_params']['coords_linear_layer'] = True
        return params
    return jobset_recurrent_brush_stroke('jobset45', 'model81', update=update)

def jobset46():

    def update(params):
        rng = random
        params['model_params']['stride'] = False
        params['model_params']['sigma'] = rng.choice((None, 1))
        params['model_params']['normalize'] = 'sigmoid'
        params['model_params']['out_reduce'] = rng.choice(('sum', 'over', 'max'))
        params['model_params']['inp_reduce'] = rng.choice(('sum', 'over', 'prev'))
        del params['model_params']['nb_recurrent_layers']
        params['model_params']['nb_recurrent_units'] = params['model_params']['nb_recurrent_units'][0]
        params['dataset'] = 'iam'
        params['force_w'] = 64
        params['force_h'] = 64
        return params
    return jobset_recurrent_brush_stroke('jobset46', 'model82', update=update)

def jobset47():

    def update(params):
        rng = random
        params['model_params']['stride'] = False
        params['model_params']['sigma'] = rng.choice((None, 1))
        params['model_params']['normalize'] = 'sigmoid'
        params['model_params']['out_reduce'] = rng.choice(('sum', 'over', 'max'))
        params['model_params']['inp_reduce'] = rng.choice(('sum', 'over', 'prev'))
        del params['model_params']['nb_recurrent_layers']
        params['model_params']['nb_recurrent_units'] = params['model_params']['nb_recurrent_units'][0]
        params['dataset'] = 'iam'
        params['force_w'] = 28
        params['force_h'] = 28
        return params
    return jobset_recurrent_brush_stroke('jobset47', 'model82', update=update)


def jobset48():

    def update(params):
        rng = random
        sigma = rng.choice((1, 0.5, 'predicted'))
        model_params = dict(
            nonlin_out='sigmoid',
            reduce_func=rng.choice(('sum', 'over', 'max')),
            normalize_func='sigmoid',
            x_sigma=sigma,
            y_sigma=sigma,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            recurrent_model=rng.choice(('gru', 'lstm', 'rnn')),
            eps=0
        )
        params['model_params'].update(model_params)
        params['dataset'] = 'aloi'
        return params

    return jobset_recurrent_brush_stroke('jobset48', 'model83', update=update)


def jobset49():

    def update(params):
        rng = random
        sigma = rng.choice((1, 0.5, 'predicted'))
        model_params = dict(
            nonlin_out='sigmoid',
            reduce_func=rng.choice(('sum', 'over', 'max')),
            normalize_func='sigmoid',
            x_sigma=sigma,
            y_sigma=sigma,
            x_stride=1,
            y_stride=1,
            patch_index=0,
            color='predicted',
            x_min=0,
            x_max='width',
            y_min=0,
            y_max='height',
            recurrent_model=rng.choice(('gru', 'lstm', 'rnn')),
            eps=0
        )
        params['model_params'].update(model_params)
        params['dataset'] = 'omniglot'
        return params

    return jobset_recurrent_brush_stroke('jobset49', 'model83', update=update)




@click.command()
@click.option('--where', default='', help='jobset name', required=False)
@click.option('--nb', default=1, help='nb of repetitions', required=False)
def insert(where, nb):
    where = globals()[where]
    total = 0
    for _ in range(nb):
        total += where()
    print("Total number of jobs added : {}".format(total))

if __name__ == '__main__':
    insert()
