from string import Template

from compatibility_core import Case, LibraryType


LIGHTGBM_VERSIONS = [
    '2.3.0',
    '2.2.3',
    '2.2.2',
    '2.2.1',
    '2.2.0',
    '2.1.2',
    '2.1.1',
    '2.1.0',
    '2.0.12',
    '2.0.11',
    '2.0.10',
]

XGBOOST_VERSIONS = [
    '0.90',
    '0.82',
    '0.72.1',
]


class BaseCase(Case):
    files = dict(
        model_filename='model.txt',
        true_predictions_filename='true_predictions.txt',
        predictions_filename='predictions.txt',
        data_filename='data.txt',
    )
    python_template=None
    go_template=None

    def compare(self):
        self.compare_matrices(
            matrix1_filename=self.files['true_predictions_filename'],
            matrix2_filename=self.files['predictions_filename'],
            tolerance=1e-10,
            max_number_of_mismatches_ratio=0.0
        )

    def go_code(self):
        return self.go_template.substitute(self.files)

    def python_code(self):
        return self.python_template.substitute(self.files)

class LGBaseCase(BaseCase):
    library = LibraryType.LIGHTGBM
    versions = LIGHTGBM_VERSIONS


class XGBaseCase(BaseCase):
    library = LibraryType.XGBOOST
    versions = XGBOOST_VERSIONS


class LGBreastCancer(LGBaseCase):
    python_template = Template("""
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
y_pred = clf.predict(X_test, raw_score=True)

clf.save_model('$model_filename')  # save the model in txt format
np.savetxt('$true_predictions_filename', y_pred)
np.savetxt('$data_filename', X_test, delimiter='\t')
""")

    go_template = Template("""
package main

import (
    "github.com/dmitryikh/leaves"
    "github.com/dmitryikh/leaves/mat"
)

func main() {
    test, err := mat.DenseMatFromCsvFile("$data_filename", 0, false, "\t", 0.0)
    if err != nil {
        panic(err)
    }

    model, err := leaves.LGEnsembleFromFile("$model_filename", false)
    if err != nil {
        panic(err)
    }
    predictions := mat.DenseMatZero(test.Rows, model.NOutputGroups())
    err = model.PredictDense(test.Values, test.Rows, test.Cols, predictions.Values, 0, 1)
    if err != nil {
        panic(err)
    }

    err = predictions.ToCsvFile("$predictions_filename", "\t")
    if err != nil {
        panic(err)
    }
}
""")


class LGIrisRandomForest(LGBaseCase):
    python_template = Template("""
import numpy as np
import pickle
from sklearn import datasets
import lightgbm as lgb
from sklearn.model_selection import train_test_split


data = datasets.load_iris()
X = data['data']
y = data['target']
y[y > 0] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

n_estimators = 30
d_train = lgb.Dataset(X_train, label=y_train)
params = {
    'boosting_type': 'rf',
    'objective': 'binary',
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'bagging_freq': 1,
}

clf = lgb.train(params, d_train, n_estimators)

y_pred = clf.predict(X_test)

model_filename = 'lg_rf_iris.model'
pred_filename = 'lg_rf_iris_true_predictions.txt'
# test_filename = 'iris_test.libsvm'

clf.save_model('$model_filename')
np.savetxt('$true_predictions_filename', y_pred)
datasets.dump_svmlight_file(X_test, y_test, '$data_filename')
""")

    go_template = Template("""
package main

import (
    "github.com/dmitryikh/leaves"
    "github.com/dmitryikh/leaves/mat"
)

func main() {
	test, err := mat.CSRMatFromLibsvmFile("$data_filename", 0, true)
	if err != nil {
		panic(err)
	}

	model, err := leaves.LGEnsembleFromFile("$model_filename", false)
	if err != nil {
		panic(err)
	}

    predictions := mat.DenseMatZero(test.Rows(), model.NOutputGroups())
	err = model.PredictCSR(test.RowHeaders, test.ColIndexes, test.Values, predictions.Values, 0, 1)
    if err != nil {
        panic(err)
    }

    err = predictions.ToCsvFile("$predictions_filename", "\t")
    if err != nil {
        panic(err)
    }
}
""")


class XGIrisMulticlass(XGBaseCase):
    python_template = Template("""
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
n_estimators = 20
clf = xgb.train(params, xg_train, n_estimators)
y_pred = clf.predict(xg_test, output_margin=True)
# save the model in binary format
clf.save_model('$model_filename')
np.savetxt('$true_predictions_filename', y_pred, delimiter='\t')
datasets.dump_svmlight_file(X_test, y_test, '$data_filename')
""")

    go_template = Template("""
package main

import (
    "github.com/dmitryikh/leaves"
    "github.com/dmitryikh/leaves/mat"
)

func main() {
	test, err := mat.CSRMatFromLibsvmFile("$data_filename", 0, true)
	if err != nil {
		panic(err)
	}

	model, err := leaves.XGEnsembleFromFile("$model_filename", false)
	if err != nil {
		panic(err)
	}

    predictions := mat.DenseMatZero(test.Rows(), model.NOutputGroups())
	err = model.PredictCSR(test.RowHeaders, test.ColIndexes, test.Values, predictions.Values, 0, 1)
    if err != nil {
        panic(err)
    }

    err = predictions.ToCsvFile("$predictions_filename", "\t")
    if err != nil {
        panic(err)
    }
}
""")

    def compare(self):
        self.compare_matrices(
            matrix1_filename=self.files['true_predictions_filename'],
            matrix2_filename=self.files['predictions_filename'],
            tolerance=1e-6,
            max_number_of_mismatches_ratio=0.0
        )


cases = [
    LGBreastCancer,
    LGIrisRandomForest,
    XGIrisMulticlass,
]
