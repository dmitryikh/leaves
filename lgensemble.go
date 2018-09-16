package leaves

import (
	"fmt"
)

// LGEnsemble is LightGBM model (ensemble of trees)
type LGEnsemble struct {
	Trees         []LGTree
	MaxFeatureIdx uint32
}

// NTrees returns number of trees in ensemble
func (e *LGEnsemble) NTrees() int {
	return len(e.Trees)
}

// Predict calculates prediction from ensembles of trees. Only `nTrees` first
// trees will be used. If `len(fvals)` is not enough function will quietly
// return 0.0. Note, that result is a raw score (before sigmoid function
// transformation and etc)
func (e *LGEnsemble) Predict(fvals []float64, nTrees int) float64 {
	if e.MaxFeatureIdx+1 > uint32(len(fvals)) {
		return 0.0
	}
	ret := 0.0
	if nTrees > 0 {
		nTrees = minInt(nTrees, e.NTrees())
	} else {
		nTrees = e.NTrees()
	}

	for i := 0; i < nTrees; i++ {
		ret += e.Trees[i].predict(fvals)
	}
	return ret
}

// PredictCSR calculates predictions from ensembles of trees. `indptr`, `cols`,
// `vals` represent data structures from Compressed Sparse Row Matrix format (see CSRMat).
// Only `nTrees` first trees will be used.
// Note, that result is a raw score (before sigmoid function transformation and etc).
// Note, `predictions` slice should be properly allocated on call side
func (e *LGEnsemble) PredictCSR(indptr []uint32, cols []uint32, vals []float64, predictions []float64, nTrees int) {
	fvals := make([]float64, e.MaxFeatureIdx+1)
	for i := 0; i < len(indptr)-1; i++ {
		start := indptr[i]
		end := indptr[i+1]
		for j := start; j < end; j++ {
			if cols[j] < uint32(len(fvals)) {
				fvals[cols[j]] = vals[j]
			}
		}
		predictions[i] = e.Predict(fvals, nTrees)
		for j := start; j < end; j++ {
			if cols[j] < uint32(len(fvals)) {
				fvals[cols[j]] = 0.0
			}
		}
	}
}

// PredictDense calculates predictions from ensembles of trees. `vals`, `rows`,
// `cols` represent data structures from Rom Major Matrix format (see DenseMat).
// Only `nTrees` first trees will be used.
// Note, that result is a raw score (before sigmoid function transformation and etc).
// Note, `predictions` slice should be properly allocated on call side
func (e *LGEnsemble) PredictDense(vals []float64, nrows uint32, ncols uint32, predictions []float64, nTrees int) error {
	if ncols == 0 || e.MaxFeatureIdx > ncols-1 {
		return fmt.Errorf("incorrect number of columns")
	}
	for i := uint32(0); i < nrows; i++ {
		predictions[i] = e.Predict(vals[i*ncols:(i+1)*ncols], nTrees)
	}
	return nil
}
