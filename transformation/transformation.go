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
)

func (t TransformType) Name() string {
	transformNames := [...]string{
		"raw",
		"logistic",
		"softmax",
	}
	if t < Raw || t > Softmax {
		return "unknown"
	}

	return transformNames[t]
}
