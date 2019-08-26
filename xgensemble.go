package leaves

import (
	"math"

	"github.com/dmitryikh/leaves/util"
)

// xgEnsemble is XGBoost model (ensemble of trees)
type xgEnsemble struct {
	Trees            []lgTree
	MaxFeatureIdx    int
	nRawOutputGroups int
	TreeInfo         []int
	BaseScore        float64
	WeightDrop       []float64
	// name contains the origin of the model (examples: 'xgboost.gbtree', 'xgboost.dart')
	name string
}

func (e *xgEnsemble) NEstimators() int {
	return len(e.Trees) / e.nRawOutputGroups
}

func (e *xgEnsemble) NRawOutputGroups() int {
	return e.nRawOutputGroups
}

func (e *xgEnsemble) NFeatures() int {
	if e.MaxFeatureIdx > 0 {
		return e.MaxFeatureIdx + 1
	}
	return 0
}

func (e *xgEnsemble) Name() string {
	return e.name
}

func (e *xgEnsemble) adjustNEstimators(nEstimators int) int {
	if nEstimators > 0 {
		nEstimators = util.MinInt(nEstimators*e.nRawOutputGroups, e.NEstimators()*e.nRawOutputGroups)
	} else {
		nEstimators = e.NEstimators() * e.nRawOutputGroups
	}
	return nEstimators
}

func (e *xgEnsemble) predictInner(fvals []float64, nEstimators int, predictions []float64, startIndex int) {
	for k := 0; k < e.nRawOutputGroups; k++ {
		predictions[startIndex+k] = e.BaseScore
		for i := 0; i < nEstimators; i++ {
			if e.TreeInfo[i] == k {
				predictions[startIndex+k] += e.Trees[i].predict(fvals) * e.WeightDrop[i]
			}
		}
	}
}

func (e *xgEnsemble) resetFVals(fvals []float64) {
	for j := 0; j < len(fvals); j++ {
		fvals[j] = math.NaN()
	}
}
