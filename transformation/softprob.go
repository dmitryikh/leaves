package transformation

import (
	"fmt"
	"math"
)

type TransformSoftprob struct {
	NClasses int
}

func (t *TransformSoftprob) Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	if len(rawPredictions) != len(outputPredictions) {
		return fmt.Errorf("expected len(rawPredictions) = %d (got %d)", t.NClasses, len(rawPredictions))
	}

	sum := 0.0

	for i, v := range rawPredictions {
		exp := math.Exp(v)
		outputPredictions[i] = exp
		sum += exp
	}

	for i := range outputPredictions {
		outputPredictions[i] /= sum
	}

	return nil
}

func (t *TransformSoftprob) NOutputGroups() int {
	return t.NClasses
}

func (t *TransformSoftprob) Type() TransformType {
	return Softprob
}

func (t *TransformSoftprob) Name() string {
	return Softprob.Name()
}
