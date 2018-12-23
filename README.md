# leaves

[![version](https://img.shields.io/badge/version-0.8.0-yellow.svg)](https://semver.org)
[![Build Status](https://travis-ci.org/dmitryikh/leaves.svg?branch=master)](https://travis-ci.org/dmitryikh/leaves)
[![GoDoc](https://godoc.org/github.com/dmitryikh/leaves?status.png)](https://godoc.org/github.com/dmitryikh/leaves)
[![Coverage Status](https://coveralls.io/repos/github/dmitryikh/leaves/badge.svg?branch=master)](https://coveralls.io/github/dmitryikh/leaves?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/dmitryikh/leaves)](https://goreportcard.com/report/github.com/dmitryikh/leaves)

![Logo](logo.png)

## Introduction

_leaves_ is a library implementing prediction code for GBRT (Gradient Boosting Regression Trees) models in **pure Go**. The goal of the project - make it possible to use models from popular GBRT frameworks in Go programs without C API bindings.

**NOTE**: Before `1.0.0` release the API is a subject to change.

## Features
  * General Features:
    * support parallel predictions for batches
    * support sigmoid, softmax transformation functions
  * Support LightGBM ([repo](https://github.com/Microsoft/LightGBM)) models:
    * read models from `text` format and from `JSON` format
    * support `gbdt`, `rf` (random forest) and `dart` models
    * support multiclass predictions
    * addition optimizations for categorical features (for example, _one hot_ decision rule)
    * addition optimizations exploiting only prediction usage
  * Support XGBoost ([repo](https://github.com/dmlc/xgboost)) models:
    * read models from binary format
    * support `gbtree`, `gblinear`, `dart` models
    * support multiclass predictions
    * support missing values (`nan`)
  * Support scikit-learn ([repo](https://github.com/scikit-learn/scikit-learn)) tree models (experimental support):
    * read models from pickle format (protocol `0`)
    * support `sklearn.ensemble.GradientBoostingClassifier`


## Usage examples

In order to start, go get this repository:

```sh
go get github.com/dmitryikh/leaves
```

Minimal example:

```go
package main

import (
	"fmt"

	"github.com/dmitryikh/leaves"
)

func main() {
	// 1. Read model
	useTransformation := true
	model, err := leaves.LGEnsembleFromFile("lightgbm_model.txt", useTransformation)
	if err != nil {
		panic(err)
	}

	// 2. Do predictions!
	fvals := []float64{1.0, 2.0, 3.0}
	p := model.PredictSingle(fvals, 0)
	fmt.Printf("Prediction for %v: %f\n", fvals, p)
}
```

In order to use XGBoost model, just change `leaves.LGEnsembleFromFile`, to `leaves.XGEnsembleFromFile`.

## Documentation

Documentation is hosted on godoc ([link](https://godoc.org/github.com/dmitryikh/leaves)). Documentation contains complex usage examples and full API reference. Some additional information about usage examples can be found in [leaves_test.go](leaves_test.go).

## Compatibility

Most _leaves_ features are tested to be compatible with old and coming versions of GBRT libraries. In [compatibility.md](compatibility.md) one can found detailed report about _leaves_ correctness against different versions of external GBRT libraries.

Some additional information on new features and backward compatibility can be found in [NOTES.md](NOTES.md).

## Benchmark

Below are comparisons of prediction speed on batches (~1000 objects in 1 API
call). Hardware: MacBook Pro (15-inch, 2017), 2,9 GHz Intel Core i7, 16 ГБ
2133 MHz LPDDR3. C API implementations were called from python bindings. But
large batch size should neglect overhead of python bindings. _leaves_
benchmarks were run by means of golang test framework: `go test -bench`. See
[benchmark](benchmark) for mode details on measurments. See
[testdata/README.md](testdata/README.md) for data preparation pipelines.

Single thread:

| Test Case | Features | Trees | Batch size |  C API  | _leaves_ |
|-----------|----------|-------|------------|---------|----------|
| LightGBM [MS LTR](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 137 |500 | 1000 | 49ms | 51ms |
| LightGBM [Higgs](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 28 | 500 | 1000 | 50ms | 50ms |
| LightGBM KDD Cup 99* | 41 | 1200 | 1000 | 70ms | 85ms |
| XGBoost Higgs | 28 | 500 | 1000 | 44ms | 50ms |

4 threads:

| Test Case | Features | Trees | Batch size |  C API  | _leaves_ |
|-----------|----------|-------|------------|---------|----------|
| LightGBM [MS LTR](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 137 |500 | 1000 | 14ms | 14ms |
| LightGBM [Higgs](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) | 28 | 500 | 1000 | 14ms | 14ms |
| LightGBM KDD Cup 99* | 41 | 1200 | 1000 | 19ms | 24ms |
| XGBoost Higgs | 28 | 500 | 1000 | ? | 14ms |

(?) - currenly I'm unable to utilize multithreading form XGBoost predictions by means of python bindings

(*) - KDD Cup 99 problem involves continuous and categorical features simultaneously

## Limitations

  * LightGBM models:
    * limited support of transformation functions (support only sigmoid, softmax)
  * XGBoost models:
    * limited support of transformation functions (support only sigmoid, softmax)
    * could be slight divergence between C API predictions vs. _leaves_ because of floating point convertions and comparisons tolerances
  * scikit-learn tree models:
    * no support transformations functions. Output scores is _raw scores_ (as from `GradientBoostingClassifier.decision_function`)
    * only pickle protocol `0` is supported
    * could be slight divergence between sklearn predictions vs. _leaves_ because of floating point convertions and comparisons tolerances

## Contacts

In case if you are interested in the project or if you have questions, please contact with me by
email: khdmitryi ```at``` gmail.com
