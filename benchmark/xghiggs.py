import logging
import numpy as np
from sklearn.datasets import load_svmlight_file
import timeit
import xgboost

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

data_filename = '../testdata/higgs_1000examples_test.libsvm'
model_filename ='../testdata/xghiggs.model'
true_pred_filename ='../testdata/xghiggs_1000examples_true_predictions.txt'

logging.info(f'start loading test data from {data_filename}')
X, _ = load_svmlight_file(data_filename, zero_based=True)
X = xgboost.DMatrix(X)
logging.info(f'load test data: {X.num_row()} x {X.num_col()}')

ytrue = np.genfromtxt(true_pred_filename)
logging.info(f'load true predictions from {true_pred_filename}')

logging.info(f'start loading model from {model_filename}')
# NOTE: it seems like I don't have control on number of threads using in predictions
# set OMP_NUM_THREADS also doesn't have any effect
xg= xgboost.Booster(model_file=model_filename, params={'nthread': 1})
logging.info('load model')

logging.info('compare predictions')
ypred = xg.predict(X, output_margin=True)

if np.allclose(ytrue, ypred):
    logging.info('predictions are valid')
else:
    logging.error('!!! wrong predictions')
    topn = 10
    for i in range(10):
        logging.error(f'{ytrue[i]} {ypred[i]}')


logging.info('start benchmark')
m = timeit.repeat('ypred = xg.predict(X, output_margin=True)', repeat=100, number=1, globals=globals())
m = np.array(m) * 1000.0
logging.info(f'done')
logging.info(f'timings (μs): min = {np.min(m):.4f}, mean = {np.mean(m):.4f}, max = {np.max(m):.4f}, std = {np.std(m):.4f}')
