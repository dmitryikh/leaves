# leaves

[![Build Status](https://travis-ci.org/dmitryikh/leaves.svg?branch=master)](https://travis-ci.org/dmitryikh/leaves)
[![Coverage Status](https://coveralls.io/repos/github/dmitryikh/leaves/badge.svg?branch=master)](https://coveralls.io/github/dmitryikh/leaves)
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

## Limitations

  * LightGBM models:
    * not supported transformations functions (sigmoid, lambdarank, etc). Output scores is _raw scores_

## Contacts

In case if you are interested in the project or if you have questions, please contact with me by
email: khdmitryi ```at``` gmail.com


    