import argparse
import lightgbm
import logging
import numpy as np
from sklearn.datasets import load_svmlight_file
import timeit

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help="data filename", type=str, required=True)
parser.add_argument('-m', '--model', help="model filename", type=str, required=True)
parser.add_argument('-t', '--true', help="filename with true predictions", type=str, required=True)
parser.add_argument('-j', help="number of threads", type=int, default=1)
params = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

logging.info(f'start loading test data from {params.data}')
if params.data.endswith('.libsvm'):
    X, _ = load_svmlight_file(params.data, zero_based=True)
elif params.data.endswith('.tsv'):
    X = np.genfromtxt(params.data, delimiter='\t')
else:
    ValueError(f"unknown data file type: 'f{params.data}''")
logging.info(f'load test data: {X.shape}')

ytrue = np.genfromtxt(params.true)
logging.info(f'load true predictions from {params.true}')

logging.info(f'start loading model from {params.model}')
lg = lightgbm.Booster(model_file=params.model, params={'num_threads': params.j})
logging.info(f'load model: {lg.num_feature()} features')

logging.info('compare predictions')
ypred = lg.predict(X, raw_score=True, num_threads=params.j)

if np.allclose(ytrue, ypred):
    logging.info('predictions are valid')
else:
    logging.error('!!! wrong predictions')
    topn = 10
    for i in range(10):
        logging.error(f'{ytrue[i]} {ypred[i]}')


logging.info('start benchmark')
m = timeit.repeat('ypred = lg.predict(X, raw_score=True, num_threads=params.j)', repeat=100, number=1, globals=globals())
m = np.array(m) * 1000.0
logging.info(f'done')
logging.info(f'timings (ms): min = {np.min(m):.4f}, mean = {np.mean(m):.4f}, max = {np.max(m):.4f}, std = {np.std(m):.4f}')
