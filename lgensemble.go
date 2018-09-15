package leaves

import (
	"fmt"
)

// LGEnsemble ..
type LGEnsemble struct {
	Trees         []LGTree
	MaxFeatureIdx uint32
}

func (e *LGEnsemble) NTrees() int {
	return len(e.Trees)
}

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

func (e *LGEnsemble) PredictDense(vals []float64, nrows uint32, ncols uint32, predictions []float64, nTrees int) error {
	if ncols == 0 || e.MaxFeatureIdx > ncols-1 {
		return fmt.Errorf("incorrect ncols")
	}
	for i := uint32(0); i < nrows; i++ {
		predictions[i] = e.Predict(vals[i*ncols:(i+1)*ncols], nTrees)
	}
	return nil
}
