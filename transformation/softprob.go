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

	for i, r := range rawPredictions {
		outputPredictions[i] = (1 / (1 + math.Exp(-r)))
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
