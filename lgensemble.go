package leaves

import (
	"github.com/dmitryikh/leaves/util"
)

// lgEnsemble is LightGBM model (ensemble of trees)
type lgEnsemble struct {
	Trees            []lgTree
	MaxFeatureIdx    int
	nRawOutputGroups int
	// lgEnsemble suits for different models from different packages (ex., LightGBM gbrt & sklearn gbrt)
	// name contains the origin of the model
	name string
	// averageOutput = true means that trees predictions should be averaged (like in random forest)
	// NOTE: LightGBM original implementation always divides result by NEstimators() if average_output set.
	// `leaves` implementation divides result by nEstimators (adjusted number of trees used for prediction)
	averageOutput bool
}

func (e *lgEnsemble) NEstimators() int {
	return len(e.Trees) / e.nRawOutputGroups
}

func (e *lgEnsemble) NRawOutputGroups() int {
	return e.nRawOutputGroups
}

func (e *lgEnsemble) NFeatures() int {
	if e.MaxFeatureIdx > 0 {
		return e.MaxFeatureIdx + 1
	}
	return 0
}

func (e *lgEnsemble) NLeaves() []int {
	nleaves := make([]int, e.NEstimators()*e.NRawOutputGroups())
	for estimatorID := 0; estimatorID < e.NEstimators(); estimatorID++ {
		for groupID := 0; groupID < e.NRawOutputGroups(); groupID++ {
			nleaves[groupID*e.NEstimators()+estimatorID] = e.Trees[estimatorID*e.NRawOutputGroups()+groupID].nLeaves()
		}
	}
	return nleaves
}

func (e *lgEnsemble) Name() string {
	return e.name
}

func (e *lgEnsemble) predictInner(fvals []float64, nEstimators int, predictions []float64, startIndex int) {
	for k := 0; k < e.nRawOutputGroups; k++ {
		predictions[startIndex+k] = 0.0
	}

	coef := 1.0
	if e.averageOutput {
		coef = 1.0 / float64(nEstimators)
	}

	for i := 0; i < nEstimators; i++ {
		for k := 0; k < e.nRawOutputGroups; k++ {
			pred, _ := e.Trees[i*e.nRawOutputGroups+k].predict(fvals)
			predictions[startIndex+k] += pred * coef
		}
	}
}

func (e *lgEnsemble) predictLeafIndicesInner(fvals []float64, nEstimators int, predictions []float64, startIndex int) {
	nResults := e.nRawOutputGroups * nEstimators
	for k := 0; k < nResults; k++ {
		predictions[startIndex+k] = 0.0
	}

	for i := 0; i < nEstimators; i++ {
		for k := 0; k < e.nRawOutputGroups; k++ {
			_, idx := e.Trees[i*e.nRawOutputGroups+k].predict(fvals)
			// note that we save leaf idx as float64 for type consistency over different types of results
			predictions[startIndex+k*nEstimators+i] = float64(idx)
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
