/*
Package leaves is pure Go implemetation of prediction part for GBRT (Gradient
Boosting Regression Trees) models from popular frameworks.

General
All loaded models exibit the same interface from `Ensemble struct`. One can
use method `Name` to get string representation of model origin. Possible name
values are "lightgbm.gbdt", "lightgbm.rf", "xgboost.gbtree", "xgboost.gblinear", etc.

LightGBM model

Example: binary classification

Python script to build the model:

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
	# note raw_score=True used here because `leaves` output only raw scores
	y_pred = clf.predict(X_test, raw_score=True)

	clf.save_model('lg_breast_cancer.model')  # save the model in txt format
	np.savetxt('lg_breast_cancer_true_predictions.txt', y_pred)
	datasets.dump_svmlight_file(X_test, y_test, 'breast_cancer_test.libsvm')

Go code to test leaves predictions on the model:

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
		model, err := leaves.LGEnsembleFromFile("lg_breast_cancer.model")
		if err != nil {
			panic(err)
		}
		fmt.Printf("Name: %s\n", model.Name())
		fmt.Printf("NFeatures: %d\n", model.NFeatures())
		fmt.Printf("NClasses: %d\n", model.NClasses())
		fmt.Printf("NEstimators: %d\n", model.NEstimators())

		// loading true predictions as DenseMat
		truePredictions, err := mat.DenseMatFromCsvFile("lg_breast_cancer_true_predictions.txt", 0, false, "\t", 0.0)
		if err != nil {
			panic(err)
		}

		// preallocate slice to store model predictions
		predictions := make([]float64, test.Rows*model.NClasses())
		// do predictions
		model.PredictDense(test.Values, test.Rows, test.Cols, predictions, 0, 1)
		// compare results
		const tolerance = 1e-6
		if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
			panic(fmt.Errorf("different predictions: %s", err.Error()))
		}
		fmt.Println("Predictions the same!")
	}

Output:

	Name: lightgbm.gbdt
	NFeatures: 30
	NClasses: 1
	NEstimators: 30
	Predictions the same!

*/
package leaves
