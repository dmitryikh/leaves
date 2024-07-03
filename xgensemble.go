package leaves

import (
	"context"
	"math"

	"github.com/manujz/leaves/util"
)

// xgEnsemble is XGBoost model (ensemble of trees)
type xgEnsemble struct {
	Trees            []lgTree
	MaxFeatureIdx    int
	nRawOutputGroups int
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

func (e *xgEnsemble) NLeaves() []int {
	nleaves := make([]int, e.NEstimators()*e.NRawOutputGroups())
	for estimatorID := 0; estimatorID < e.NEstimators(); estimatorID++ {
		for groupID := 0; groupID < e.NRawOutputGroups(); groupID++ {
			nleaves[groupID*e.NEstimators()+estimatorID] = e.Trees[estimatorID*e.NRawOutputGroups()+groupID].nLeaves()
		}
	}
	return nleaves
}

func (e *xgEnsemble) Name() string {
	return e.name
}

func (e *xgEnsemble) adjustNEstimators(nEstimators int) int {
	if nEstimators > 0 {
		nEstimators = util.MinInt(nEstimators, e.NEstimators())
	} else {
		nEstimators = e.NEstimators()
	}
	return nEstimators
}

func (e *xgEnsemble) predictInner(ctx context.Context, fvals []float64, nEstimators int, predictions []float64, startIndex int) error {
	for k := 0; k < e.nRawOutputGroups; k++ {
		predictions[startIndex+k] = e.BaseScore
	}

	for i := 0; i < nEstimators; i++ {
		for k := 0; k < e.nRawOutputGroups; k++ {
			ID := i*e.nRawOutputGroups + k
			pred, _, err := e.Trees[ID].predict(ctx, fvals)
			if err != nil {
				return err
			}
			predictions[startIndex+k] += pred * e.WeightDrop[ID]
		}
	}

	return nil
}

func (e *xgEnsemble) predictLeafIndicesInner(ctx context.Context, fvals []float64, nEstimators int, predictions []float64, startIndex int) error {
	nResults := e.nRawOutputGroups * nEstimators
	for k := 0; k < nResults; k++ {
		predictions[startIndex+k] = 0.0
	}

	for i := 0; i < nEstimators; i++ {
		for k := 0; k < e.nRawOutputGroups; k++ {
			_, idx, err := e.Trees[i*e.nRawOutputGroups+k].predict(ctx, fvals)
			if err != nil {
				return err
			}
			// note that we save leaf idx as float64 for type consistency over different types of results
			predictions[startIndex+k*nEstimators+i] = float64(idx)
		}
	}

	return nil
}

func (e *xgEnsemble) resetFVals(fvals []float64) {
	for j := 0; j < len(fvals); j++ {
		fvals[j] = math.NaN()
	}
}
