package transformation

type Transform interface {
	Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error
	NOutputGroups() int
	Type() TransformType
	Name() string
}

// TransformType is enum for various transformation functions that could be
// applied to the raw model results.
type TransformType int

const (
	// Raw is a TransformType that do nothing
	Raw TransformType = 0
	// Logistic is a TransformType that apply logistic function in order to obtain
	// positive class probabilities
	Logistic TransformType = 1
	// Softmax is a TransformType to obtain multiclass probabilities
	Softmax TransformType = 2
	// LeafIndex is a TransformType to return leaf indices from decision trees in ensemble
	LeafIndex TransformType = 3

	Last TransformType = 3
)

func (t TransformType) Name() string {
	transformNames := [...]string{
		"raw",
		"logistic",
		"softmax",
		"leaf_index",
	}
	if t < Raw || t > Last {
		return "unknown"
	}

	return transformNames[t]
}
