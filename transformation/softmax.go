package transformation

import (
	"fmt"

	"github.com/dmitryikh/leaves/util"
)

type TransformSoftmax struct {
	NClasses int
}

func (t *TransformSoftmax) Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	if len(rawPredictions) != t.NClasses {
		return fmt.Errorf("expected len(rawPredictions) = %d (got %d)", t.NClasses, len(rawPredictions))
	}

	util.SoftmaxFloat64Slice(rawPredictions, outputPredictions, startIndex)
	return nil
}

func (t *TransformSoftmax) NOutputGroups() int {
	return t.NClasses
}

func (t *TransformSoftmax) Type() TransformType {
	return Softmax
}

func (t *TransformSoftmax) Name() string {
	return Softmax.Name()
}
