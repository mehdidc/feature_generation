import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')
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

import forestci as fci

from tools.common import to_generation, to_training

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


class BayesianRandomForest(object):

    def __init__(self, rf_model):
        self.rf_model = rf_model

    def fit(self, X, y):
        self.X_train = X
        return self.rf_model.fit(X, y)
    
    def predict(self, X, return_std=False):
        mu = self.rf_model.predict(X)
        if return_std:
            inbag = fci.calc_inbag(self.X_train.shape[0], self.rf_model)
            var = fci.random_forest_error(self.rf_model, inbag, self.X_train, X)
            std = np.sqrt(var)
            return mu, std
        else:
            return mu

def get_scores_bandit(inputs, outputs, new_inputs=None, algo='thompson'):
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import Imputer
    from hp_toolkit.bandit import Thompson, UCB, BayesianOptimization, expected_improvement
    from hp_toolkit.helpers import Pipeline
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import r2_score
    import types

    preprocess = lambda x:frozendict(flatten_dict(x))
    inputs = map(preprocess, inputs)
    new_inputs = map(preprocess, new_inputs)
    #reg = GaussianProcessRegressor(normalize_y=True)
    #reg = LinearRegression()
    #reg = RandomForestRegressor() 
    reg = BayesianRandomForest(RandomForestRegressor())
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
    elif algo == 'ei':
        bandit = BayesianOptimization(model, criterion=expected_improvement)
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
        jobs = to_generation(jobs, db=db)
    func = partial(db.get_value, field=col)
    outputs = map(func, jobs)
    return outputs

if __name__ == '__main__':
    from lightjob.db import SUCCESS
    X, y = get_hypers(where='jobset67', state=SUCCESS)
    scores = get_scores_thompson(X, y, new_inputs=X)
    print(scores)
