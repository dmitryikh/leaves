# Data preparation

## Higgs dataset for XGBoost model

  1. clone https://github.com/guolinke/boosting_tree_benchmarks
  2. to download raw data and prepare datasets follow instructions in https://github.com/guolinke/boosting_tree_benchmarks/blob/master/data/readme.md
  3. run script in boosting_tree_benchmarks/xgboost
  ```sh
    head -n 1000 ../data/higgs.test > ../data/higgs_1000examples_test.libsvm
    xgboost xgboost.conf max_bin=255 tree_method=hist grow_policy=lossguide max_depth=0 max_leaves=255 data="../data/higgs.train" eval[test]="../data/higgs.test" objective="binary:logistic" eval_metric=auc model_out=xghiggs.model 2>&1 | tee xgboost_hist_higgs_accuracy.log
    xgboost xgboost.conf task=pred model_in=xghiggs.model pred_margin=true test_path="../data/higgs_1000examples_test.libsvm" name_pred="xghiggs_1000examples_true_raw_predictions.txt"
    xgboost xgboost.conf task=pred model_in=xghiggs.model test_path="../data/higgs_1000examples_test.libsvm" name_pred="xghiggs_1000examples_true_predictions.txt"
    cp ../data/higgs_1000examples_test.libsvm $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp xghiggs_1000examples_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp xghiggs_1000examples_true_raw_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp xghiggs.model $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
  ```

## Higgs dataset for LightGBM model

  1. clone https://github.com/guolinke/boosting_tree_benchmarks
  2. to download raw data and prepare datasets follow instructions in https://github.com/guolinke/boosting_tree_benchmarks/blob/master/data/readme.md
  3. run script in boosting_tree_benchmarks/lightgbm
  ```sh
    head -n 1000 ../data/higgs.test > ../data/higgs_1000examples_test.libsvm
    lightgbm config=lightgbm.conf data=../data/higgs.train output_model=lghiggs.model objective=binary
    lightgbm task=predict data=../data/higgs_1000examples_test.libsvm input_model=lghiggs.model output_result=lghiggs_1000examples_true_predictions.txt predict_raw_score=true
    cp ../data/higgs_1000examples_test.libsvm $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp lghiggs_1000examples_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp lghiggs.model $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
  ```

## MSLTR dateset for LightGBM model

  1. clone https://github.com/guolinke/boosting_tree_benchmarks
  2. to download raw data and prepare datasets follow instructions in https://github.com/guolinke/boosting_tree_benchmarks/blob/master/data/readme.md
  3. run script in boosting_tree_benchmarks/lightgbm
  ```sh
    head -n 1000 ../data/msltr.test > ../data/msltr_1000examples_test.libsvm
    lightgbm config=lightgbm.conf data=../data/msltr.train output_model=lgmsltr.model objective=lambdarank
    lightgbm task=predict data=../data/msltr_1000examples_test.libsvm input_model=lgmsltr.model output_result=lgmsltr_1000examples_true_predictions.txt predict_raw_score=true
    cp ../data/msltr_1000examples_test.libsvm $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp lgmsltr_1000examples_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp lgmsltr.model $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
  ```

## Agaricus dataset for XGBoost model

  1. clone https://github.com/dmlc/xgboost
  2. cd to xgboost/demo/guide-python/
  3. run script there:
  ```python
    #!/usr/bin/python
    import numpy as np
    import xgboost as xgb

    ### load data in do training
    dtrain = xgb.DMatrix('../data/agaricus.txt.train')
    dtest = xgb.DMatrix('../data/agaricus.txt.test')
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 3
    bst = xgb.train(param, dtrain, num_round, watchlist)

    ypred = bst.predict(dtest, output_margin=True)
    np.savetxt('xgagaricus_true_predictions.txt', ypred)
    bst.save_model('xgagaricus.model')
  ```
  4.
  ```sh
    cp xgagaricus_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp xgagaricus.model $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp ../data/agaricus.txt.test $GOPATH/src/github.com/dmitryikh/leaves/testdata/agaricus_test.libsvm
  ```

## Agaricus dataset for XGBoost gblinear model

  1. clone https://github.com/dmlc/xgboost
  2. cd to xgboost/demo/guide-python/
  3. run script there:
  ```python
    import numpy as np
    import xgboost as xgb

    ### load data in do training
    dtrain = xgb.DMatrix('../data/agaricus.txt.train')
    dtest = xgb.DMatrix('../data/agaricus.txt.test')
    param = {'booster': 'gblinear', 'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 3
    bst = xgb.train(param, dtrain, num_round, watchlist)

    ypred = bst.predict(dtest)
    ypred_raw = bst.predict(dtest, output_margin=True)
    np.savetxt('xgblin_agaricus_true_predictions.txt', ypred, delimiter='\t')
    np.savetxt('xgblin_agaricus_true_raw_predictions.txt', ypred_raw, delimiter='\t')
    bst.save_model('xgblin_agaricus.model')
  ```
  4.
  ```sh
    cp xgblin_agaricus_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp xgblin_agaricus_true_raw_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp xgblin_agaricus.model $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp ../data/agaricus.txt.test $GOPATH/src/github.com/dmitryikh/leaves/testdata/agaricus_test.libsvm
  ```

## Agaricus dataset for XGBoost DART model

  1. clone https://github.com/dmlc/xgboost
  2. cd to xgboost/demo/guide-python/
  3. run script there:
  ```python
  import numpy as np
  import xgboost as xgb
  # read in data
  dtrain = xgb.DMatrix('../data/agaricus.txt.train')
  dtest = xgb.DMatrix('../data/agaricus.txt.test')
  param = {
      'booster': 'gbtree',
      'max_depth': 5, 'learning_rate': 0.1,
      'objective': 'binary:logistic', 'silent': True,
      'sample_type': 'uniform',
      'normalize_type': 'tree',
      'rate_drop': 0.1,
      'skip_drop': 0.5
  }
  num_round = 20
  bst = xgb.train(param, dtrain, num_round)

  # make prediction
  ypred = bst.predict(dtest, output_margin=True, ntree_limit=10)
  np.savetxt('xg_dart_agaricus_true_predictions.txt', ypred)
  bst.save_model('xg_dart_agaricus.model')
  ```
  4.
  ```sh
    cp xg_dart_agaricus_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp xg_dart_agaricus.model $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp ../data/agaricus.txt.test $GOPATH/src/github.com/dmitryikh/leaves/testdata/agaricus_test.libsvm
  ```

## Multiclass classification dataset for LightGBM model
  1. clone https://github.com/Microsoft/LightGBM
  2. cd to examples/multiclass_classification
  3. run
  ```sh
  lightgbm boosting_type=gbdt objective=multiclass num_class=5 max_bin=255 data=multiclass.train num_trees=10 learning_rate=0.05 num_leaves=31 output_model=lgmulticlass.model
  lightgbm input_model=lgmulticlass.model data=multiclass.test task=predict output_result=lgmulticlass_true_predictions.txt
  lightgbm input_model=lgmulticlass.model data=multiclass.test task=predict output_result=lgmulticlass_true_raw_predictions.txt predict_raw_score=true
  cp multiclass.test $GOPATH/src/github.com/dmitryikh/leaves/testdata/multiclass_test.tsv
  cp lgmulticlass.model $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
  cp lgmulticlass_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
  cp lgmulticlass_true_raw_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
  ```


## Dermatology dataset for XGBoost
  1. clone https://github.com/dmlc/xgboost
  2. demo/multiclass_classification
  3. Read instruction how to take data (https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data)
  4. run (modified train.py)
  ```python
from __future__ import division

import numpy as np
import xgboost as xgb
import sklearn.datasets as ds

# label need to be 0 to num_class -1
data = np.loadtxt('./dermatology.data', delimiter=',',
        converters={33: lambda x:int(x == '?'), 34: lambda x:int(x) - 1})
sz = data.shape

train = data[:int(sz[0] * 0.7), :]
test = data[int(sz[0] * 0.7):, :]
train_X = train[:, :33]
train_Y = train[:, 34]
test_X = test[:, :33]
test_Y = test[:, 34]

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 6

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist)

ypred = bst.predict(xg_test, output_margin=True).reshape(test_Y.shape[0], 6)
np.savetxt('xgdermatology_true_predictions.txt', ypred, delimiter='\t')
bst.save_model('xgdermatology.model')
ds.dump_svmlight_file(test_X, test_Y, 'dermatology_test.libsvm')
  ```
  5.
  ```sh
    cp xgdermatology_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp xgdermatology.model $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp dermatology_test.libsvm $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
  ```


## Gradient Boosting Classifier for scikit-learn
  1. run
  ```sh
    cd internal/pickle/testdata
    python gradient_boosting_classifier.py
  ```


## Iris for scikit-learn
  1. run
  ```sh
    cd testdata
    python iris.py
  ```


## Iris for LightGBM Random Forest
  1. run
  ```sh
    cd pytotestdata
    python iris_lightgbm_rf.py
  ```

## Breast Cancer for LightGBM DART model (+ JSON format model)
  1. run
  ```sh
    cd testdata
    python lg_dart_breast_cancer.py
  ```

## KDD Cup 99 for LightGBM model
  1. run
  ```sh
    cd testdata
    python lg_kddcup99.py
  ```

## KDD Cup 99 for LightGBM model for benchmark
  1. run
  ```sh
    cd testdata
    python lg_kddcup99.py bench
  ```
