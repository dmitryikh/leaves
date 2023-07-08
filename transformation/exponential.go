package transformation

import (
	"fmt"
	"math"
)

type TransformExponential struct{}

func (t *TransformExponential) Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	if len(rawPredictions) != 1 {
		return fmt.Errorf("expected len(rawPredictions) = 1 (got %d)", len(rawPredictions))
	}

	outputPredictions[startIndex] = math.Exp(rawPredictions[0])
	return nil
}

func (t *TransformExponential) NOutputGroups() int {
	return 1
}

func (t *TransformExponential) Type() TransformType {
	return Exponential
}

func (t *TransformExponential) Name() string {
	return Logistic.Name()
}

