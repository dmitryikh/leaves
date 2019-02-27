/*
Package leaves is pure Go implemetation of prediction part for GBRT (Gradient
Boosting Regression Trees) models from popular frameworks.

General
All loaded models exibit the same interface from `Ensemble struct`. One can
use method `Name` to get string representation of model origin. Possible name
values are "lightgbm.gbdt", "lightgbm.rf", "xgboost.gbtree", "xgboost.gblinear", etc.

LightGBM model

Example: binary classification

build_breast_cancer_model.py:

	import lightgbm as lgb
	import numpy as np
	from sklearn import datasets
	from sklearn.model_selection import train_test_split

	X, y = datasets.load_breast_cancer(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	n_estimators = 30
	d_train = lgb.Dataset(X_train, label=y_train)
	params = {
	    'boosting_type': 'gbdt',
	    'objective': 'binary',
	}
	clf = lgb.train(params, d_train, n_estimators)
	y_pred = clf.predict(X_test)
	y_pred_raw = clf.predict(X_test, raw_score=True)

	clf.save_model('lg_breast_cancer.model')  # save the model in txt format
	np.savetxt('lg_breast_cancer_true_predictions.txt', y_pred)
	np.savetxt('lg_breast_cancer_true_predictions_raw.txt', y_pred_raw)
	np.savetxt('breast_cancer_test.tsv', X_test, delimiter='\t')

predict_breast_cancer_model.go:

	package main

	import (
		"fmt"

		"github.com/dmitryikh/leaves"
		"github.com/dmitryikh/leaves/mat"
		"github.com/dmitryikh/leaves/util"
	)

	func main() {
		// loading test data
		test, err := mat.DenseMatFromCsvFile("breast_cancer_test.tsv", 0, false, "\t", 0.0)
		if err != nil {
			panic(err)
		}

		// loading model
		model, err := leaves.LGEnsembleFromFile("lg_breast_cancer.model", true)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Name: %s\n", model.Name())
		fmt.Printf("NFeatures: %d\n", model.NFeatures())
		fmt.Printf("NOutputGroups: %d\n", model.NOutputGroups())
		fmt.Printf("NEstimators: %d\n", model.NEstimators())
		fmt.Printf("Transformation: %s\n", model.Transformation().Name())

		// loading true predictions as DenseMat
		truePredictions, err := mat.DenseMatFromCsvFile("lg_breast_cancer_true_predictions.txt", 0, false, "\t", 0.0)
		if err != nil {
			panic(err)
		}
		truePredictionsRaw, err := mat.DenseMatFromCsvFile("lg_breast_cancer_true_predictions_raw.txt", 0, false, "\t", 0.0)
		if err != nil {
			panic(err)
		}

		// preallocate slice to store model predictions
		predictions := make([]float64, test.Rows*model.NOutputGroups())
		// do predictions
		model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)
		// compare results
		const tolerance = 1e-6
		if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
			panic(fmt.Errorf("different predictions: %s", err.Error()))
		}

		// compare raw predictions (before transformation function)
		rawModel := model.EnsembleWithRawPredictions()
		rawModel.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)
		if err := util.AlmostEqualFloat64Slices(truePredictionsRaw.Values, predictions, tolerance); err != nil {
			panic(fmt.Errorf("different raw predictions: %s", err.Error()))
		}
		fmt.Println("Predictions the same!")
	}

Output:

	Name: lightgbm.gbdt
	NFeatures: 30
	NOutputGroups: 1
	NEstimators: 30
	Transformation: logistic
	Predictions the same!

XGBoost Model

example: Multiclass Classification

build_iris_model.py

	import numpy as np
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	import xgboost as xgb

	X, y = datasets.load_iris(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	xg_train = xgb.DMatrix(X_train, label=y_train)
	xg_test = xgb.DMatrix(X_test, label=y_test)
	params = {
	    'objective': 'multi:softmax',
	    'num_class': 3,
	}
	n_estimators = 5
	clf = xgb.train(params, xg_train, n_estimators)
	# use output_margin=True because of `leaves` predictions are raw scores (before
	# transformation function)
	y_pred = clf.predict(xg_test, output_margin=True)
	# save the model in binary format
	clf.save_model('xg_iris.model')
	np.savetxt('xg_iris_true_predictions.txt', y_pred, delimiter='\t')
	datasets.dump_svmlight_file(X_test, y_test, 'iris_test.libsvm')

predict_iris_model.go:

	package main

	import (
		"fmt"

		"github.com/dmitryikh/leaves"
		"github.com/dmitryikh/leaves/mat"
		"github.com/dmitryikh/leaves/util"
	)

	func main() {
		// loading test data
		csr, err := mat.CSRMatFromLibsvmFile("iris_test.libsvm", 0, true)
		if err != nil {
			panic(err)
		}

		// loading model
		model, err := leaves.XGEnsembleFromFile("xg_iris.model", false)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Name: %s\n", model.Name())
		fmt.Printf("NFeatures: %d\n", model.NFeatures())
		fmt.Printf("NOutputGroups: %d\n", model.NOutputGroups())
		fmt.Printf("NEstimators: %d\n", model.NEstimators())

		// loading true predictions as DenseMat
		truePredictions, err := mat.DenseMatFromCsvFile("xg_iris_true_predictions.txt", 0, false, "\t", 0.0)
		if err != nil {
			panic(err)
		}

		// preallocate slice to store model predictions
		predictions := make([]float64, csr.Rows()*model.NOutputGroups())
		// do predictions
		model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, 1)
		// compare results
		const tolerance = 1e-6
		// compare results. Count number of mismatched values beacase of floating point
		// tolerances in decision rule
		mismatch, err := util.NumMismatchedFloat64Slices(truePredictions.Values, predictions, tolerance)
		if err != nil {
			panic(err)
		}
		if mismatch > 2 {
			panic(fmt.Errorf("mismatched more than %d predictions", mismatch))
		}
		fmt.Printf("Predictions the same! (mismatch = %d)\n", mismatch)
	}

Output:

	Name: xgboost.gbtree
	NFeatures: 4
	NOutputGroups: 3
	NEstimators: 5
	Predictions the same! (mismatch = 0)

Notes on XGBoost DART support

Please note that one must not provide nEstimators = 0 when predict with DART models from xgboost. For more details see xgboost's documentation.

Notes on LightGBM DART support

Models trained with 'boosting_type': 'dart' options can be loaded with func `leaves.LGEnsembleFromFile`.
But the name of the model (given by `Name()` method) will be 'lightgbm.gbdt', because LightGBM model format doesn't distinguish 'gbdt' and 'dart' models.

*/
package leaves
