import lightgbm as lgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

n_estimators = 10
d_train = lgb.Dataset(X_train, label=y_train)
params = {
    'boosting_type': 'dart',
    'objective': 'binary',
}
clf = lgb.train(params, d_train, n_estimators)
y_pred = clf.predict(X_test)

clf.save_model('lg_dart_breast_cancer.model')  # save the model in txt format
np.savetxt('lg_dart_breast_cancer_true_predictions.txt', y_pred)
np.savetxt('breast_cancer_test.tsv', X_test, delimiter='\t')
d = clf.dump_model()
import json
with open('lg_dart_breast_cancer.json', 'w') as fout:
    json.dump(d, fout, indent=1)
