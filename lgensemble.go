package leaves

import (
	"github.com/dmitryikh/leaves/util"
)

// lgEnsemble is LightGBM model (ensemble of trees)
type lgEnsemble struct {
	Trees         []lgTree
	MaxFeatureIdx int
	nClasses      int
}

func (e *lgEnsemble) NEstimators() int {
	return len(e.Trees) / e.nClasses
}

func (e *lgEnsemble) NClasses() int {
	return e.nClasses
}

func (e *lgEnsemble) NFeatures() int {
	if e.MaxFeatureIdx > 0 {
		return e.MaxFeatureIdx + 1
	}
	return 0
}

func (e *lgEnsemble) Name() string {
	return "lightgbm"
}

func (e *lgEnsemble) predictInner(fvals []float64, nEstimators int, predictions []float64, startIndex int) {
	for k := 0; k < e.nClasses; k++ {
		predictions[startIndex+k] = 0.0
	}
	for i := 0; i < nEstimators; i++ {
		for k := 0; k < e.nClasses; k++ {
			predictions[startIndex+k] += e.Trees[i*e.nClasses+k].predict(fvals)
		}
	}
}

func (e *lgEnsemble) adjustNEstimators(nEstimators int) int {
	if nEstimators > 0 {
		nEstimators = util.MinInt(nEstimators, e.NEstimators())
	} else {
		nEstimators = e.NEstimators()
	}
	return nEstimators
}

func (e *lgEnsemble) resetFVals(fvals []float64) {
	for j := 0; j < len(fvals); j++ {
		fvals[j] = 0.0
	}
}
