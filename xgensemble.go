package leaves

import (
	"math"

	"github.com/dmitryikh/leaves/util"
)

// xgEnsemble is XGBoost model (ensemble of trees)
type xgEnsemble struct {
	Trees         []lgTree
	MaxFeatureIdx int
	nClasses      int
	TreeInfo      []int
	BaseScore     float64
}

func (e *xgEnsemble) NEstimators() int {
	return len(e.Trees) / e.nClasses
}

func (e *xgEnsemble) NClasses() int {
	return e.nClasses
}

func (e *xgEnsemble) NFeatures() int {
	if e.MaxFeatureIdx > 0 {
		return e.MaxFeatureIdx + 1
	}
	return 0
}

func (e *xgEnsemble) Name() string {
	return "gbtree"
}

func (e *xgEnsemble) adjustNEstimators(nEstimators int) int {
	if nEstimators > 0 {
		nEstimators = util.MinInt(nEstimators*e.nClasses, e.NEstimators()*e.nClasses)
	} else {
		nEstimators = e.NEstimators() * e.nClasses
	}
	return nEstimators
}

func (e *xgEnsemble) predictInner(fvals []float64, nEstimators int, predictions []float64, startIndex int) {
	for k := 0; k < e.nClasses; k++ {
		predictions[startIndex+k] = e.BaseScore
		for i := 0; i < nEstimators; i++ {
			if e.TreeInfo[i] == k {
				predictions[startIndex+k] += e.Trees[i].predict(fvals)
			}
		}
	}
}

func (e *xgEnsemble) resetFVals(fvals []float64) {
	for j := 0; j < len(fvals); j++ {
		fvals[j] = math.NaN()
	}
}
