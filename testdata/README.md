# Data preparation

## Higgs dataset for XGBoost model

  1. clone https://github.com/guolinke/boosting_tree_benchmarks
  2. to download raw data and prepare datasets follow instructions in https://github.com/guolinke/boosting_tree_benchmarks/blob/master/data/readme.md
  3. run script in boosting_tree_benchmarks/xgboost
  ```sh
    head -n 1000 ../data/higgs.test > ../data/higgs_1000examples_test.libsvm
    xgboost xgboost.conf max_bin=255 tree_method=hist grow_policy=lossguide max_depth=0 max_leaves=255 data="../data/higgs.train" eval[test]="../data/higgs.test" objective="binary:logistic" eval_metric=auc model_out=xghiggs.model 2>&1 | tee xgboost_hist_higgs_accuracy.log
    xgboost xgboost.conf task=pred model_in=higgs.model pred_margin=true test_path="../data/higgs_1000examples_test.libsvm" name_pred="xghiggs_1000examples_true_predictions.txt"
    cp ../data/higgs_1000examples_test.libsvm $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
    cp xghiggs_1000examples_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
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

  ## Multiclass classification dataset for LightGBM model
  1. clone https://github.com/Microsoft/LightGBM
  2. cd to examples/multiclass_classification
  3. run
  ```sh
  lightgbm boosting_type=gbdt objective=multiclass num_class=5 max_bin=255 data=multiclass.train num_trees=10 learning_rate=0.05 num_leaves=31 output_model=lgmulticlass.model
  lightgbm input_model=lgmulticlass.model data=multiclass.test task=predict output_result=lgmulticlass_true_predictions.txt predict_raw_score=true
  cp multiclass.test $GOPATH/src/github.com/dmitryikh/leaves/testdata/multiclass_test.tsv
  cp lgmulticlass.model $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
  cp lgmulticlass_true_predictions.txt $GOPATH/src/github.com/dmitryikh/leaves/testdata/.
  ```