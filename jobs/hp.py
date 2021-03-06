import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')
from collections import defaultdict
from collections import Iterable, Mapping
from hyperopt import hp, Trials
from hyperopt import fmin, tpe, rand
from hyperopt.pyll.stochastic import sample
import numpy as np
from collections import OrderedDict

from hp_toolkit.helpers import flatten_dict, DictFlattener
from hp_toolkit.bandit import Thompson

from functools import partial

import forestci as fci

from tools.common import to_generation, to_training, store, retrieve
from frozendict import frozendict

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
    
    def sample_y(self, X, random_state=None):
        rng = np.random.RandomState(random_state)
        mu = self.rf_model.predict(X)
        inbag = fci.calc_inbag(self.X_train.shape[0], self.rf_model)
        var = fci.random_forest_error(self.rf_model, inbag, self.X_train, X)
        return rng.multivariate_normal(mu, np.diag(var))

class Transformer(object):

    def __init__(self, func):
        self.func = func
        
    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y=y)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.func(X)

def get_scores_bandit(inputs, outputs, new_inputs=None, algo='thompson'):
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import Imputer
    from hp_toolkit.bandit import Thompson, UCB, BayesianOptimization, expected_improvement, Simple
    from hp_toolkit.helpers import Pipeline
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import r2_score
    import types
    preprocess = lambda x:frozendict(flatten_dict(x))
    inputs = map(preprocess, inputs)
    new_inputs = map(preprocess, new_inputs)
    #reg = RandomForestRegressor()
    reg =  BayesianRandomForest(RandomForestRegressor())
    model = Pipeline(DictVectorizer(sparse=False), Imputer(), reg)
    algos = {'thompson': Thompson, 'simple': Simple, 'ucb': UCB, 'ei': partial(BayesianOptimization, criterion=expected_improvement)}
    cls = algos[algo]
    bandit = cls(model)
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
    func = partial(db.get_value, field=col, if_not_found=np.nan)
    outputs = map(func, jobs)
    return outputs

class History(object):
    
    def __init__(self):
        self.hist = defaultdict(list)
    
    def save(self, filename):
        store(self.hist, filename)

    def load(self, filename):
        self.hist = retrieve(filename)

    def push(self,data, label):
        self.hist[label].append(data)

if __name__ == '__main__':
    from lightjob.db import SUCCESS
    X, y = get_hypers(where='jobset67', state=SUCCESS)
    scores = get_scores_bandit(X, y, new_inputs=X, algo='simple')
    r = History()
    r.push(X, 'hi')
