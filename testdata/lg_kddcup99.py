import lightgbm as lgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys

if len(sys.argv) == 2 and sys.argv[1] == 'bench':
    for_bench = True
else:
    for_bench = False

data = datasets.fetch_kddcup99(subset='SA')
X, y = data['data'], data['target']

# feature description from http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
# 0 : duration: continuous.
# 1 : protocol_type: symbolic.
# 2 : service: symbolic.
# 3 : flag: symbolic.
# 4 : src_bytes: continuous.
# 5 : dst_bytes: continuous.
# 6 : land: symbolic.
# 7 : wrong_fragment: continuous.
# 8 : urgent: continuous.
# 9 : hot: continuous.
# 10: num_failed_logins: continuous.
# 11: logged_in: symbolic.
# 12: num_compromised: continuous.
# 13: root_shell: continuous.
# 14: su_attempted: continuous.
# 15: num_root: continuous.
# 16: num_file_creations: continuous.
# 17: num_shells: continuous.
# 18: num_access_files: continuous.
# 19: num_outbound_cmds: continuous.
# 20: is_host_login: symbolic.
# 21: is_guest_login: symbolic.
# 22: count: continuous.
# 23: srv_count: continuous.
# 24: serror_rate: continuous.
# 25: srv_serror_rate: continuous.
# 26: rerror_rate: continuous.
# 27: srv_rerror_rate: continuous.
# 28: same_srv_rate: continuous.
# 29: diff_srv_rate: continuous.
# 30: srv_diff_host_rate: continuous.
# 31: dst_host_count: continuous.
# 32: dst_host_srv_count: continuous.
# 33: dst_host_same_srv_rate: continuous.
# 34: dst_host_diff_srv_rate: continuous.
# 35: dst_host_same_src_port_rate: continuous.
# 36: dst_host_srv_diff_host_rate: continuous.
# 37: dst_host_serror_rate: continuous.
# 38: dst_host_srv_serror_rate: continuous.
# 39: dst_host_rerror_rate: continuous.
# 40: dst_host_srv_rerror_rate: continuous.

symbolic_features = [1, 2, 3]
# some fatures added as categorical (despite their nature) for test purpose
categorical_features = symbolic_features + [6, 7, 8, 9, 10, 11, 13, 20, 21, 22, 23]

# convert symbolic features to numerical
y = LabelEncoder().fit_transform(y)
for idx in symbolic_features:
    X[:, idx] = LabelEncoder().fit_transform(X[:, idx])

test_size = 1000 if for_bench else 0.005
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

n_estimators = 100 if for_bench else 10
d_train = lgb.Dataset(X_train, label=y_train)
params = {
    'boosting_type': 'gbrt',
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),
}
clf = lgb.train(params, d_train, n_estimators, categorical_feature=categorical_features)
y_pred = clf.predict(X_test, raw_score=True)

suffix = '_for_bench' if for_bench else ''
clf.save_model(f'lg_kddcup99{suffix}.model')  # save the model in txt format
np.savetxt(f'lg_kddcup99_true_predictions{suffix}.txt', y_pred, delimiter='\t')
# NOTE: fmt tuned to get small size file
np.savetxt(f'kddcup99_test{suffix}.tsv', X_test, delimiter='\t', fmt='%0.18g')
