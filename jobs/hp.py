from hyperopt import hp, Trials
from hyperopt import fmin, tpe, rand


def linearize_dict(d):
    return linearize_('', d, [])


def linearize_(k, v, path):
    if not isinstance(v, dict):
        return {'_'.join(path): v}
    res = {}
    for klocal, vlocal in v.items():
        res.update(linearize_(klocal, vlocal, path + [klocal]))
    return res


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
            if inputs[feval_.i] != x:
                raise ValueError(
                    "The {}-th element of the provided inputs do not"
                    "correspond to the inputs asked by hyperopt"
                    ": {} vs {}".format(feval_.i + 1, inputs[feval_.i], x))
            output = outputs[feval_.i]
            feval_.i += 1
            if feval_.i == len(inputs) and feval.next_seed:
                feval.rng.randint(0, feval.next_seed)
            return output
        else:
            return feval(x)
    feval_.i = 0
    return feval_


def get_next_hyperopt(inputs, outputs, space,
                      algo='tpe', rstate=None, next_seed=None):
    # dummy func
    def fn(x):
        fn.next_val = x
        return 1
    fn.next_seed = next_seed
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


if __name__ == '__main__':
    import numpy as np
    inputs = []
    space = {'x': hp.uniform('x', 0, 1)}
    outputs = []
    for i in range(10):
        rng = np.random.RandomState(123)
        next_hp = get_next_hyperopt(
            inputs, outputs, space, algo='rand', rstate=rng)
        inputs.append(next_hp)
        outputs.append((next_hp['x'] - 2) ** 2)
        print(next_hp)
