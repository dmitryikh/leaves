import numpy as np
np.random.seed(10)

import os
import pickle

from sklearn.datasets import make_classification, dump_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

n_estimator = 50
max_depth = 3
X, y = make_classification(n_samples=10000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

grd = GradientBoostingClassifier(n_estimators=n_estimator, max_depth=max_depth)
grd.fit(X_train, y_train)

def print_estimator(e):
    thresholds = e.tree_.threshold
    features = e.tree_.feature
    impurity = e.tree_.impurity
    children_right = e.tree_.children_right
    children_left = e.tree_.children_left
    n_node_samples = e.tree_.n_node_samples
    weighted_n_node_samples = e.tree_.weighted_n_node_samples
    print("trueNodes := []SklearnNode{")
    for i in range(len(thresholds)):
        print(f"\tSklearnNode{{LeftChild: {children_left[i]}, RightChild: {children_right[i]}, "
              f"Feature: {features[i]}, Threshold: {thresholds[i]:.17g}, Impurity: {impurity[i]:.17g}, "
              f"NNodeSamples: {n_node_samples[i]}, WeightedNNodeSamples: {weighted_n_node_samples[i]:.17g}}},")
    print("}")

# print decision_tree class solo
tree_filename = 'decision_tree_regressor.pickle0'
pickle.dump(grd.estimators_[0][0], open(tree_filename, 'wb'), protocol=0)
print_estimator(grd.estimators_[0][0])

# print whoile model and it's RAW predictions
base = os.path.join('..', '..', '..', 'testdata')
model_filename = os.path.join(base, 'sk_gradient_boosting_classifier.model')
pred_filename = os.path.join(base, 'sk_gradient_boosting_classifier_true_predictions.txt')
test_filename = os.path.join(base, 'sk_gradient_boosting_classifier_test.libsvm')

pickle.dump(grd, open(model_filename, 'wb'), protocol=0)
y_pred_grd = grd.decision_function(X_test)
np.savetxt(pred_filename, y_pred_grd)
dump_svmlight_file(X_test, y_test, test_filename)