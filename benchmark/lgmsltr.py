import lightgbm
import logging
import numpy as np
from sklearn.datasets import load_svmlight_file
import timeit

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

data_filename = '../testdata/msltr_1000examples_test.libsvm'
model_filename ='../testdata/lgmsltr.model'
true_pred_filename ='../testdata/lgmsltr_1000examples_true_predictions.txt'

logging.info(f'start loading test data from {data_filename}')
X, _ = load_svmlight_file(data_filename, zero_based=True)
logging.info(f'load test data: {X.shape}')

ytrue = np.genfromtxt(true_pred_filename)
logging.info(f'load true predictions from {true_pred_filename}')

logging.info(f'start loading model from {model_filename}')
lg = lightgbm.Booster(model_file=model_filename, params={'num_threads': 1})
logging.info(f'load model: {lg.num_feature()} features')

logging.info('compare predictions')
ypred = lg.predict(X, raw_score=True, num_threads=4)

if np.allclose(ytrue, ypred):
    logging.info('predictions are valid')
else:
    logging.error('!!! wrong predictions')
    topn = 10
    for i in range(10):
        logging.error(f'{ytrue[i]} {ypred[i]}')


logging.info('start benchmark')
m = timeit.repeat('ypred = lg.predict(X, raw_score=True, num_threads=1)', repeat=100, number=1, globals=globals())
m = np.array(m) * 1000.0
logging.info(f'done')
logging.info(f'timings (ms): min = {np.min(m):.4f}, mean = {np.mean(m):.4f}, max = {np.max(m):.4f}, std = {np.std(m):.4f}')
