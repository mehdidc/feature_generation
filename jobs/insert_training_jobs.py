import json
from collections import OrderedDict
from lightjob.utils import summarize
from lightjob.db import SUCCESS
from hp import get_next_hyperopt
from hyperopt import hp

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
    crit = 'knn_classification_accuracy'
    model = 'model64'

    def filter_func(job):
        if job['stats'] != SUCCESS:
            return False
        if 'stats' not in job:
            return False
        if crit not in job['stats']:
            return False
        if job['content']['model_name'] != model:
            return False
        return True
    jobs = db.jobs_filter(filter_func)
    inputs = [j['content'] for j in jobs]
    outputs = [j['stats'][crit] for j in jobs]
    print('len of prior params : {}'.format(len(inputs)))

    model_params_space = OrderedDict(
        use_wta_sparse=hp.choice('use_wta_sparse', (True, False)),
        wta_sparse_perc=hp.uniform('wta_sparse_perc', 0, 1),
        nb_layers=1 + hp.randint('nb_layers', 5),
        nb_hidden_units=100 + hp.randint('nb_hidden_units', 2000))
    space = OrderedDict(
        model_params=model_params_space,
        denoise=hp.choice('denoise', (hp.uniform('denoise_pr', 0, 1), None)),
        noise=hp.choice('noise', ('zero_masking', 'salt_and_pepper')),
        walkback=1 + hp.randint('walkback', 5),
        walkback_mode='bengio_without_sampling',
        autoencoding_loss='squared_error',
        mode='minibatch',
        contractive=False,
        contractive_coef=None,
        marginalized=False,
        binarize_thresh=hp.choice('binarize_thresh', (None, 0.5)),
        eval_stats=[crit]
    )
    seed = np.random.randint(0, 99999999)
    rng = np.random.RandomState(seed)
    params = get_next_hyperopt(
        inputs,
        outputs,
        space,
        algo='rand',
        rstate=rng)
    budget_hours = 4
    model_name = 'model64'
    dataset = 'digits'
    jobset_name = "jobset34"
    cmd = build_cmd(model_name=model_name,
                    dataset=dataset,
                    params=params,
                    budget_hours=budget_hours)
    nb = job_write(params, cmd, where=jobset_name)
    return nb


def jobset35():
    rng = random
    nb_layers = rng.randint(1, 7)
    nb_filters = [2 ** rng.randint(5, 9) for _ in range(nb_layers)]
    model_params = OrderedDict(
        nb_layers=nb_layers,
        nb_filters=nb_filters,
        filter_size=rng.choice((3, 5)),
        use_channel=rng.choice((True, False)),
        use_spatial=rng.choice((True, False)),
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
    budget_hours = 10
    model_name = 'model73'
    dataset = 'digits'
    jobset_name = "jobset35"
    cmd = build_cmd(model_name=model_name,
                    dataset=dataset,
                    params=params,
                    budget_hours=budget_hours)
    nb = job_write(params, cmd, where=jobset_name)
    return nb


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
