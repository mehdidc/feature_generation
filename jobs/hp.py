from collections import Iterable, Mapping
from hyperopt import hp, Trials
from hyperopt import fmin, tpe, rand
from hyperopt.pyll.stochastic import sample
from skopt.space import Categorical, Dimension
import numpy as np
from collections import OrderedDict
from skopt import gp_minimize, forest_minimize

from hp_toolkit.helpers import flatten_dict, DictFlattener
from hp_toolkit.bandit import Thompson

from frozendict import frozendict
from functools import partial
def deep_frozendict(x):
    if isinstance(x, Mapping):
        return frozendict({k: deep_frozendict(v) for k, v in x.items()})
    else:
        return x

def linearize_dict(d):
    return linearize_('', d, [])

def linearize_(k, v, path):
    if not isinstance(v, dict):
        return {'.'.join(path): v}
    res = {}
    for klocal, vlocal in v.items():
        res.update(linearize_(klocal, vlocal, path + [klocal]))
    return res

def delinearize_dict(d, t=OrderedDict):
    d_out = t()
    for k, v in d.items():
        d_out_ = d_out
        for s in k.split('.'):
            d_out_[s] = t()
            d_out_ = d_out_[s]
    for k, v in d.items():
        d_out_ = d_out
        path = k.split('.')
        for s in path[0:-1]:
            d_out_ = d_out_[s]
        d_out_[path[-1]] = v
    return d_out


def feed(feval, inputs, outputs):
    """
    decorate the hyperopt feval (fmin) function
    by a collection of pairs of inputs and outputs at the beginning.
    the motivation is to have a way to save hyperopt 'state' and load
    it later to continue later.
    """
    def feval_(x):
        if feval_.i < len(inputs):
            if inputs[feval_.i] != x:
                raise ValueError(
                    "The {}-th element of the provided inputs do not"
                    "correspond to the inputs asked by hyperopt"
                    ": {} vs {}".format(feval_.i + 1, inputs[feval_.i], x))
            output = outputs[feval_.i]
            feval_.i += 1
            return output
        else:
            return feval(x)
    feval_.i = 0
    return feval_


def get_next_hyperopt(inputs, outputs, space,
                      algo='tpe', rstate=None):
    # dummy func
    def fn(x):
        fn.next_val = x
        return 1
    fn.rng = rstate if rstate is not None else np.random

    if algo == 'tpe':
        algo = tpe.suggest
    elif algo == 'rand':
        algo = rand.suggest
    else:
        raise Exception('Expected tpe or rand, got : {}'.format(algo))
    trials = Trials()
    fmin(feed(fn, inputs, outputs),
         space,
         algo=algo,
         max_evals=len(inputs) + 1,
         trials=trials,
         rstate=rstate)
    return fn.next_val


def get_from_trials(trials, name):
    return [t[name] for t in trials.trials]

def main():
    import numpy as np
    inputs = []
    space = {'x': hp.uniform('x', 0, 1)}
    trials = Trials()
    def fn(s):
        return s['x']
    rng = np.random.RandomState(5)
    fmin(fn, space, algo=tpe.suggest, max_evals=2, rstate=rng, trials=trials)
    print(get_from_trials(trials, 'result'))
    rng = np.random.RandomState(5)
    fmin(fn, space, algo=tpe.suggest, max_evals=3, rstate=rng, trials=trials)
    print(get_from_trials(trials, 'result'))
    outputs = []
    for i in range(10):
        rng = np.random.RandomState(123)
        next_hp = get_next_hyperopt(
            inputs, outputs, space, algo='rand', rstate=rng)
        inputs.append(next_hp)
        outputs.append((next_hp['x'] - 2) ** 2)
        print(next_hp)

def encode_dict(x, structure=None, accept_val=lambda x:True):
    if structure is None:
        x = linearize_dict(x)
        vals = x.values()
        return vals
    else:
        structure_lin = linearize_dict(structure)
        x = linearize_dict(x)
        structure_lin = recur_update(structure_lin, x, accept_val=accept_val)
        vals = structure_lin.values()
        return vals

def recur_update(d, u, accept_val=lambda v:True):
    d = d.copy()
    for k, v in u.iteritems():
        if isinstance(v, Mapping):
            if k in d:
                if accept_val(d[k]):
                    r = recur_update(d.get(k, {}), v)
                    d[k] = r
                else:
                    del d[k]
        else:
            if k in d:
                if accept_val(d[k]):
                    d[k] = u[k]
                else:
                    del d[k]
    return d

def decode_dict(x, structure=None, accept_val=lambda v:True):
    if structure is None:
        x = delinearize_dict(x)
    else:
        structure_lin = linearize_dict(structure)
        for (k, _), v in zip(structure_lin.items(), x):
            structure_lin[k] = v
        print(structure_lin)
        x = structure_lin
        x = delinearize_dict(x)
        x = recur_update(structure, x, accept_val=accept_val)

    return x


class Identity:
    """Identity transform."""

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, Xt):
        return Xt

class Constant(Dimension):

    def __init__(self, val):
        self.val = val

    def rvs(self, n_samples=1, random_state=None):
        return [self.val] * n_samples

def get_next_skopt(inputs, outputs, space, rstate=None):
    def func(x):
        func.x = x
        return 1
    res = gp_minimize(func, space, n_calls=1, n_random_starts=0, x0=inputs, y0=outputs, random_state=rstate)
    return func.x

#http://code.activestate.com/recipes/577555-object-wrapper-class/
class Wrapper(object):
    '''
    Object wrapper class.
    This a wrapper for objects. It is initialiesed with the object to wrap
    and then proxies the unhandled getattribute methods to it.
    Other classes are to inherit from it.
    '''
    def __init__(self, obj):
        '''
        Wrapper constructor.
        @param obj: object to wrap
        '''
        # wrap the object
        self._wrapped_obj = obj

    def __getattr__(self, attr):
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recurrsion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        # proxy to the wrapped object
        return getattr(self._wrapped_obj, attr)

def get_scores_bandit(inputs, outputs, new_inputs=None, algo='thompson'):
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import Imputer
    from hp_toolkit.bandit import Thompson, UCB
    from hp_toolkit.helpers import Pipeline
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import r2_score
    import types

    preprocess = lambda x:frozendict(flatten_dict(x))
    inputs = map(preprocess, inputs)
    new_inputs = map(preprocess, new_inputs)
    #reg = GaussianProcessRegressor(normalize_y=True)
    #reg = LinearRegression()
    reg = RandomForestRegressor() 
    class WrapEstimator(Wrapper):
        def fit(self, X, y):
            return self._wrapped_obj.fit(X, y)
        def predict(self, X, *args, **kwargs):
            return self._wrapped_obj.predict(X, *args, **kwargs)
    reg = WrapEstimator(reg)
    
    model = Pipeline(DictVectorizer(sparse=False), Imputer(), reg)
    if algo == 'thompson':
        bandit = Thompson(model)
    elif algo == 'ucb':
        bandit = UCB(model)
    bandit.update(inputs, outputs)
    score = r2_score(outputs, bandit.model.predict(inputs))
    print('r2 score : {}'.format(score))
    return bandit.get_action_scores(new_inputs)

def get_hypers(y_col='stats.training.avg_loss_train_fix', **kw):
    from lightjob.cli import load_db
    db = load_db()
    jobs = db.jobs_with(**kw)
    inputs = [j['content'] for j in jobs]
    outputs = get_col(jobs, y_col, db=db) 
    return inputs, outputs

def get_col(jobs, col, db):
    if col.startswith('g#'):
        col = col[2:]
        S = set(j['summary'] for j in jobs)
        ref_jobs = db.jobs_with(type='generation')
        ref_jobs = {j['content']['model_summary']:j for j in ref_jobs if j['content']['model_summary'] in S}
        jobs = map(lambda s:ref_jobs[s], S)
    func = partial(db.get_value, field=col)
    outputs = map(func, jobs)
    return outputs

if __name__ == '__main__':
    from lightjob.db import SUCCESS
    X, y = get_hypers(where='jobset67', state=SUCCESS)
    scores = get_scores_thompson(X, y, new_inputs=X)
    print(scores)
