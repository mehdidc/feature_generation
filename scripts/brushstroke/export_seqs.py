from docopt import docopt
import numpy as np
from common import load_model, build_encode_func, minibatcher 

def main():
    doc = """
    Usage:
        export_seqs.py [--from-json=JSON] [--id=ID] FILE [OUTPUT]

    Arguments:
        FILE input file
        OUTPUT output file [default:out.npz]

    Options:
        -h --help     Show this screen
        --from-json=JSON json configuration file
        --id=ID id of job
    """
    args = docopt(doc)
    model, data, layers = load_model(args['FILE'])
    encode = build_encode_func(layers) # transforms image to sequence of coordinates
    X = data.train.X if hasattr(data, 'train') else data.X
    X = model.preprocess(X)
    codes = minibatcher(encode, batchsize=1000)(X)
    print('Saving to {}'.format(args['OUTPUT']))
    np.savez(args['OUTPUT'], codes=codes)

if __name__ == '__main__':
    main()
