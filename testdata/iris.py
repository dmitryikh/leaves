import numpy as np
import pickle
from sklearn import ensemble, datasets
from sklearn.model_selection import train_test_split

data = datasets.load_iris()
X = data['data']
y = data['target']

n_estimators = 30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

gb = ensemble.GradientBoostingClassifier(n_estimators=n_estimators, random_state=0)
gb.fit(X_train, y_train)

# print whoile model and it's RAW predictions
model_filename = 'sk_iris.model'
pred_filename = 'sk_iris_true_predictions.txt'
test_filename = 'iris_test.libsvm'

pickle.dump(gb, open(model_filename, 'wb'), protocol=0)
y_pred_grd = gb.decision_function(X_test)
np.savetxt(pred_filename, y_pred_grd, delimiter='\t')
datasets.dump_svmlight_file(X_test, y_test, test_filename)