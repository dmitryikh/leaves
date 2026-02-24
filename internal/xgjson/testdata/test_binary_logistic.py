# /// script
# dependencies = ["xgboost>=2.0", "numpy", "scikit-learn"]
# ///
"""
Generates binary:logistic XGBoost test model.
Run with: uv run test_binary_logistic.py
Outputs: test_binary_logistic.json, test_binary_logistic.ubj, test_binary_logistic_expected.json
"""

import json
import os
import numpy as np
from sklearn.datasets import make_classification
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    X, y = make_classification(
        n_samples=1000,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    X_train, X_test = X[:800], X[800:]
    y_train = y[:800]

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    json_path = os.path.join(SCRIPT_DIR, "test_binary_logistic.json")
    ubj_path = os.path.join(SCRIPT_DIR, "test_binary_logistic.ubj")
    model.get_booster().save_model(json_path)
    model.get_booster().save_model(ubj_path)
    print(f"Saved {json_path}")
    print(f"Saved {ubj_path}")

    test_rows = X_test[:5]
    dtest = xgb.DMatrix(test_rows)
    raw_preds = model.get_booster().predict(dtest, output_margin=True)
    prob_preds = model.predict_proba(test_rows)[:, 1]

    expected = []
    for i in range(len(test_rows)):
        expected.append(
            {
                "features": test_rows[i].tolist(),
                "raw_score": float(raw_preds[i]),
                "probability": float(prob_preds[i]),
            }
        )

    expected_path = os.path.join(SCRIPT_DIR, "test_binary_logistic_expected.json")
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Saved {expected_path}")

    print(f"\nXGBoost version: {xgb.__version__}")
    for e in expected:
        print(f"  features={[round(v,4) for v in e['features']]}  raw={e['raw_score']:.6f}  prob={e['probability']:.6f}")


if __name__ == "__main__":
    main()
