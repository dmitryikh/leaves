# XGBoost Test Model Data

This directory contains generated XGBoost model files used by the `xgjson` package tests.

## Files

- `train.py` — Python script that generates the test models
- `test1.json` — XGBoost model in JSON format (XGBoost 2.x/3.x)
- `test1.ubj` — XGBoost model in Universal Binary JSON format (XGBoost 2.x/3.x)
- `test1_expected.json` — Input feature vectors with expected raw scores and probabilities

## Regenerating

Requires [uv](https://docs.astral.sh/uv/).

```sh
cd internal/xgjson/testdata
go generate
```

This trains a small XGBoost binary classifier (10 estimators, max_depth=3) on synthetic data
with 3 features, then writes the three output files above. Commit all three output files.
