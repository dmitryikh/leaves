# /// script
# dependencies = ["xgboost>=2.0", "numpy", "scikit-learn"]
# ///
"""
Generates multi:softprob XGBoost test model (3-class).
Run with: uv run test_multiclass.py
Outputs: test_multiclass.json, test_multiclass.ubj, test_multiclass_expected.json

NOTE on base_score: XGBoost 3.x stores a per-class base_score vector for multi-class models.
The leaves Go library (xgEnsemble) only supports a single scalar BaseScore, so it uses 0.0
for multi-class. Expected values here are the raw tree sums (output_margin minus per-class
base_score) and the softmax of those sums, which is what the Go library actually computes.
"""

import json
import os
import numpy as np
from sklearn.datasets import make_classification
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    X, y = make_classification(
        n_samples=600,
        n_features=4,
        n_classes=3,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )

    X_train, X_test = X[:500], X[500:]
    y_train = y[:500]

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    json_path = os.path.join(SCRIPT_DIR, "test_multiclass.json")
    ubj_path = os.path.join(SCRIPT_DIR, "test_multiclass.ubj")
    model.get_booster().save_model(json_path)
    model.get_booster().save_model(ubj_path)
    print(f"Saved {json_path}")
    print(f"Saved {ubj_path}")

    # Parse per-class base_scores from the saved model.
    # XGBoost 3.x stores these as "[v0,v1,...,vN]" in learner_model_param.base_score.
    with open(json_path) as f:
        saved = json.load(f)
    bs_str = saved["learner"]["learner_model_param"]["base_score"].strip("[]")
    base_scores = np.array([float(x) for x in bs_str.split(",")])
    print(f"Per-class base_scores: {base_scores.tolist()}")

    test_rows = X_test[:5]
    dtest = xgb.DMatrix(test_rows)
    # output_margin=True gives per-class raw scores including per-class base_score
    raw_preds = model.get_booster().predict(dtest, output_margin=True)  # shape: [n, 3]

    # Go uses BaseScore=0 for multi-class (xgEnsemble doesn't support per-class base_score).
    # Subtract per-class base_scores to get the pure tree sums that Go computes.
    go_raw_preds = raw_preds - base_scores  # broadcast over rows
    go_probs = softmax(go_raw_preds)

    expected = []
    for i in range(len(test_rows)):
        expected.append(
            {
                "features": test_rows[i].tolist(),
                "raw_scores": go_raw_preds[i].tolist(),
                "probabilities": go_probs[i].tolist(),
            }
        )

    expected_path = os.path.join(SCRIPT_DIR, "test_multiclass_expected.json")
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Saved {expected_path}")

    print(f"\nXGBoost version: {xgb.__version__}")
    for e in expected:
        print(f"  raw={[round(v,4) for v in e['raw_scores']]}  prob={[round(v,4) for v in e['probabilities']]}")


if __name__ == "__main__":
    main()
