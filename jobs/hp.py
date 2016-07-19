from hyperopt import hp, Trials
from hyperopt import fmin, tpe, rand

def linearize_dict(d):
    def linearize_(k, v, path):
        if not isinstance(v, dict):
            return {'.'.join(path + [k]): v}
        res = {}
        for klocal, vlocal in v.items():
            res.update(linearize_(klocal, vlocal, path + [klocal]))
        return res


def linearize_(k, v, path):
    if not isinstance(v, dict):
        return {'_'.join(path): v}
    res = {}
    for klocal, vlocal in v.items():
        res.update(linearize_(klocal, vlocal, path + [klocal]))
    return res

def linearize_dict(d):
    return linearize_('', d, [])

def delinearize_dict(d):
    d_out = {}
    for k, v in d.items():
        d_out_ = d_out
        for s in k.split('_'):
            d_out_[s] = {}
            d_out_ = d_out_[s]
    for k, v in d.items():
        d_out_ = d_out
        path = k.split('_')
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
            assert inputs[feval_.i] == x, 'Check the'
            output = outputs[feval_.i]
            feval_.i += 1
            return output
        else:
            return feval(x)
    feval_.i = 0
    return feval_

def get_next_hyperopt(inputs, outputs, space, algo='tpe'):
    def fn(x):
        fn.next_val = x
        return 1
    trials = Trials()  
    fmin(feed(fn, inputs, outputs), 
         space, 
         algo=tpe.suggest if algo=='tpe' else rand.suggest, 
         max_evals=len(inputs) + 1, 
         trials=trials,
         rseed=1)
    return fn.next_val

def get_from_trials(trials, name):
    return [t[name] for t in trials.trials]


if __name__ == '__main__':
    space = {'x': hp.uniform('x', 1, 10)}
    inputs = []
    outputs = []

    for i in range(1000):
        next_hp = get_next_hyperopt(inputs, outputs, space, algo='rand')
        inputs.append(next_hp)
        outputs.append((next_hp['x'] - 2) ** 2)
        print(next_hp)