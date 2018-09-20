package leaves

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// XGEnsemble is XGBoost model (ensemble of trees)
type XGEnsemble struct {
	Trees         []lgTree
	MaxFeatureIdx int
}

// NTrees returns number of trees in ensemble
func (e *XGEnsemble) NTrees() int {
	return len(e.Trees)
}

// PredictSingle calculates prediction from ensembles of trees. Only `nTrees` first
// trees will be used. If `len(fvals)` is not enough function will quietly
// return 0.0.
// Note, that result is a raw score (before sigmoid function transformation and etc).
// Note, nan feature values treated as missing values
func (e *XGEnsemble) PredictSingle(fvals []float64, nTrees int) float64 {
	if e.MaxFeatureIdx+1 > len(fvals) {
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

// Predict for multiclass predictions.
// NOTE: currently XGEnsemble doesn't support multiclass prediction, thus
// Predict behaves the same as PredictSingle (slightly different signatures)
func (e *XGEnsemble) Predict(fvals []float64, nTrees int, predictions []float64) error {
	if len(predictions) < 1 {
		return fmt.Errorf("predictions slice to short (should be at least %d)", 1)
	}
	if e.MaxFeatureIdx+1 > len(fvals) {
		return fmt.Errorf("incorrect number of features (%d)", len(fvals))
	}
	if nTrees > 0 {
		nTrees = minInt(nTrees, e.NTrees())
	} else {
		nTrees = e.NTrees()
	}

	predictions[0] = 0.0
	for i := 0; i < nTrees; i++ {
		predictions[0] += e.Trees[i].predict(fvals)
	}
	return nil
}

// PredictCSR calculates predictions from ensembles of trees. `indptr`, `cols`,
// `vals` represent data structures from Compressed Sparse Row Matrix format (see CSRMat).
// Only `nTrees` first trees will be used. `nThreads` points to number of
// threads that will be utilized (maximum is GO_MAX_PROCS)
// Note, that result is a raw score (before sigmoid function transformation and etc).
// Note, `predictions` slice should be properly allocated on call side
func (e *XGEnsemble) PredictCSR(indptr []int, cols []int, vals []float64, predictions []float64, nTrees int, nThreads int) error {
	nRows := len(indptr) - 1
	if nRows <= BatchSize || nThreads == 0 || nThreads == 1 {
		fvals := make([]float64, e.MaxFeatureIdx+1)
		for i := range fvals {
			fvals[i] = math.NaN()
		}
		e.predictCSRInner(indptr, cols, vals, 0, len(indptr)-1, predictions, nTrees, fvals)
		return nil
	}
	if nThreads > runtime.GOMAXPROCS(0) || nThreads < 1 {
		nThreads = runtime.GOMAXPROCS(0)
	}
	nBatches := int(math.Ceil(float64(nRows) / BatchSize))
	if nThreads > nBatches {
		nThreads = nBatches
	}
	tasks := make(chan int)

	wg := sync.WaitGroup{}
	for i := 0; i < nThreads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			fvals := make([]float64, e.MaxFeatureIdx+1)
			for i := range fvals {
				fvals[i] = math.NaN()
			}
			for {
				startIndex, more := <-tasks
				if !more {
					return
				}
				endIndex := startIndex + BatchSize
				if endIndex > nRows {
					endIndex = nRows
				}
				e.predictCSRInner(indptr, cols, vals, startIndex, endIndex, predictions, nTrees, fvals)
			}
		}()
	}

	// feed the queue
	for i := 0; i < nBatches; i++ {
		tasks <- i * BatchSize
	}
	close(tasks)
	wg.Wait()
	return nil
}

func (e *XGEnsemble) predictCSRInner(indptr []int, cols []int, vals []float64, startIndex int, endIndex int, predictions []float64, nTrees int, fvals []float64) {
	for i := startIndex; i < endIndex; i++ {
		start := indptr[i]
		end := indptr[i+1]
		for j := start; j < end; j++ {
			if cols[j] < len(fvals) {
				fvals[cols[j]] = vals[j]
			}
		}
		predictions[i] = e.PredictSingle(fvals, nTrees)
		for j := start; j < end; j++ {
			if cols[j] < len(fvals) {
				fvals[cols[j]] = math.NaN()
			}
		}
	}
}

// PredictDense calculates predictions from ensembles of trees. `vals`, `rows`,
// `cols` represent data structures from Rom Major Matrix format (see DenseMat).
// Only `nTrees` first trees will be used. `nThreads` points to number of
// threads that will be utilized (maximum is GO_MAX_PROCS)
// Note, that result is a raw score (before sigmoid function transformation and etc).
// Note, `predictions` slice should be properly allocated on call side
func (e *XGEnsemble) PredictDense(vals []float64, nrows int, ncols int, predictions []float64, nTrees int, nThreads int) error {
	nRows := nrows
	if ncols == 0 || e.MaxFeatureIdx > ncols-1 {
		return fmt.Errorf("incorrect number of columns")
	}
	if nRows <= BatchSize || nThreads == 0 || nThreads == 1 {
		// single thread calculations
		for i := 0; i < nRows; i++ {
			predictions[i] = e.PredictSingle(vals[i*int(ncols):(i+1)*int(ncols)], nTrees)
		}
		return nil
	}
	if nThreads > runtime.GOMAXPROCS(0) || nThreads < 1 {
		nThreads = runtime.GOMAXPROCS(0)
	}
	nBatches := int(math.Ceil(float64(nRows) / BatchSize))
	if nThreads > nBatches {
		nThreads = nBatches
	}
	tasks := make(chan int)

	wg := sync.WaitGroup{}
	for i := 0; i < nThreads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				startIndex, more := <-tasks
				if !more {
					return
				}
				endIndex := startIndex + BatchSize
				if endIndex > nRows {
					endIndex = nRows
				}
				for i := startIndex; i < endIndex; i++ {
					predictions[i] = e.PredictSingle(vals[i*int(ncols):(i+1)*int(ncols)], nTrees)
				}
			}
		}()
	}

	// feed the queue
	for i := 0; i < nBatches; i++ {
		tasks <- i * BatchSize
	}
	close(tasks)
	wg.Wait()
	return nil
}
