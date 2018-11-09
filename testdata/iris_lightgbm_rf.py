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

clf.save_model(model_filename)
np.savetxt(pred_filename, y_pred)
# datasets.dump_svmlight_file(X_test, y_test, test_filename)