# leaves

[![Build Status](https://travis-ci.org/dmitryikh/leaves.svg?branch=master)](https://travis-ci.org/dmitryikh/leaves)
[![Coverage Status](https://coveralls.io/repos/github/dmitryikh/leaves/badge.svg?branch=master)](https://coveralls.io/github/dmitryikh/leaves?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/dmitryikh/leaves)](https://goreportcard.com/report/github.com/dmitryikh/leaves)

![Logo](logo.png)

## Intoduction

_leaves_ is a library implementing prediction code for GBRT (Gradient Boosting Regression Trees) models in **pure Go**. The goal of the project - make it possible to use models from popular GBRT frameworks in Go programs without C API bindings.

## Features

  * Support LightGBM ([repo](https://github.com/Microsoft/LightGBM)) models:
    * reading models from text format
    * supporting numerical & categorical features
    * addition optimizations for categorical features (for example, _one hot_ decision rule)
    * addition optimizations exploiting only prediction usage
  * Support XGBoost ([repo](https://github.com/dmlc/xgboost)) models:
    * reading models from binary format
    * supporting missing values (`nan`)


## Usage examples

In order to start, go get this repository:

```sh
go get github.com/dmitryikh/leaves
```

Minimal example:

```go
package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/dmitryikh/leaves"
)

func main() {
	// 1. Open file
	path := "lightgbm_model.txt"
	reader, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer reader.Close()

	// 2. Read LightGBM model
	model, err := leaves.LGEnsembleFromReader(bufio.NewReader(reader))
	if err != nil {
		panic(err)
	}

	// 3. Do predictions!
	fvals := []float64{1.0, 2.0, 3.0}
	p := model.Predict(fvals, 0)
	fmt.Printf("Prediction for %v: %f\n", fvals, p)
}
```

In order to use XGBoost model, just change `leaves.LGEnsembleFromReader`, to `leaves.XGEnsembleFromReader`. For mode usage examples see [leaves_test.go](leaves_test.go).

## Benchmark

Below are comparisons of prediction speed on batches (~1000 objects in 1
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
| XGBoost Higgs | 28 | 500 | 1000 | 44ms | 50ms |

## Limitations

  * LightGBM models:
    * no support transformations functions (sigmoid, lambdarank, etc). Output scores is _raw scores_
  * XGBoost models:
    * no support transformations functions. Output scores is _raw scores_
	* support only `gbtree` models (most common)

## Contacts

In case if you are interested in the project or if you have questions, please contact with me by
email: khdmitryi ```at``` gmail.com
