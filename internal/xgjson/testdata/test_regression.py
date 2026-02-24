# /// script
# dependencies = ["xgboost>=2.0", "numpy", "scikit-learn"]
# ///
"""
Generates reg:squarederror XGBoost test model.
Run with: uv run test_regression.py
Outputs: test_regression.json, test_regression.ubj, test_regression_expected.json
"""

import json
import os
import numpy as np
from sklearn.datasets import make_regression
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    X, y = make_regression(n_samples=500, n_features=4, random_state=42)

    X_train, X_test = X[:400], X[400:]
    y_train = y[:400]

    model = xgb.XGBRegressor(
        n_estimators=10,
        max_depth=3,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    json_path = os.path.join(SCRIPT_DIR, "test_regression.json")
    ubj_path = os.path.join(SCRIPT_DIR, "test_regression.ubj")
    model.get_booster().save_model(json_path)
    model.get_booster().save_model(ubj_path)
    print(f"Saved {json_path}")
    print(f"Saved {ubj_path}")

    test_rows = X_test[:5]
    dtest = xgb.DMatrix(test_rows)
    # identity link: output_margin == predict()
    raw_preds = model.get_booster().predict(dtest, output_margin=True)

    expected = []
    for i in range(len(test_rows)):
        expected.append(
            {
                "features": test_rows[i].tolist(),
                "raw_score": float(raw_preds[i]),
            }
        )

    expected_path = os.path.join(SCRIPT_DIR, "test_regression_expected.json")
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Saved {expected_path}")

    print(f"\nXGBoost version: {xgb.__version__}")
    for e in expected:
        print(f"  features={[round(v,4) for v in e['features']]}  raw={e['raw_score']:.6f}")


if __name__ == "__main__":
    main()
