package leaves

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/dmitryikh/leaves/util"
)

// XGEnsemble is XGBoost model (ensemble of trees)
type XGEnsemble struct {
	Trees         []lgTree
	MaxFeatureIdx int
	nClasses      int
	TreeInfo      []int
	BaseScore     float64
}

// NTrees returns number of trees in ensemble
func (e *XGEnsemble) NTrees() int {
	return len(e.Trees)
}

// NClasses returns number of classes to predict
func (e *XGEnsemble) NClasses() int {
	return e.nClasses
}

func (e *XGEnsemble) adjustNTrees(nTrees int) int {
	if nTrees > 0 {
		nTrees = util.MinInt(nTrees*e.nClasses, e.NTrees())
	} else {
		nTrees = e.NTrees()
	}
	return nTrees
}

// PredictSingle calculates prediction from ensembles of trees. Only `nTrees` first
// trees will be used. If `len(fvals)` is not enough function will quietly
// return 0.0.
// Note, that result is a raw score (before sigmoid function transformation and etc).
// Note, nan feature values treated as missing values
func (e *XGEnsemble) PredictSingle(fvals []float64, nTrees int) float64 {
	if e.nClasses != 1 {
		return 0.0
	}
	if e.MaxFeatureIdx+1 > len(fvals) {
		return 0.0
	}
	ret := e.BaseScore
	nTrees = e.adjustNTrees(nTrees)

	for i := 0; i < nTrees; i++ {
		ret += e.Trees[i].predict(fvals)
	}
	return ret
}

func (e *XGEnsemble) predictInner(fvals []float64, nIterations int, predictions []float64, startIndex int) {
	for k := 0; k < e.nClasses; k++ {
		predictions[startIndex+k] = e.BaseScore
		for i := 0; i < nIterations; i++ {
			if e.TreeInfo[i] == k {
				predictions[startIndex+k] += e.Trees[i].predict(fvals)
			}
		}
	}
}

// Predict calculates single prediction for one or multiclass ensembles.
// Only `nTrees` first trees will be used. Note, that result is a raw score (before
// sigmoid function transformation and etc)
// NOTE: for single class predictions one can use simplified function PredictSingle
func (e *XGEnsemble) Predict(fvals []float64, nTrees int, predictions []float64) error {
	if len(predictions) < 1 {
		return fmt.Errorf("predictions slice too short (should be at least %d)", 1)
	}
	if e.MaxFeatureIdx+1 > len(fvals) {
		return fmt.Errorf("incorrect number of features (%d)", len(fvals))
	}

	nTrees = e.adjustNTrees(nTrees)
	e.predictInner(fvals, nTrees, predictions, 0)
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
	if len(predictions) < e.nClasses*nRows {
		return fmt.Errorf("predictions slice too short (should be at least %d)", e.nClasses*nRows)
	}
	nTrees = e.adjustNTrees(nTrees)
	if nRows <= BatchSize || nThreads == 0 || nThreads == 1 {
		// single thread calculations
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
			for startIndex := range tasks {
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
		e.predictInner(fvals, nTrees, predictions, i*e.nClasses)
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
	if len(predictions) < e.nClasses*nRows {
		return fmt.Errorf("predictions slice too short (should be at least %d)", e.nClasses*nRows)
	}
	if ncols == 0 || e.MaxFeatureIdx > ncols-1 {
		return fmt.Errorf("incorrect number of columns")
	}
	nTrees = e.adjustNTrees(nTrees)
	if nRows <= BatchSize || nThreads == 0 || nThreads == 1 {
		// single thread calculations
		for i := 0; i < nRows; i++ {
			e.predictInner(vals[i*int(ncols):(i+1)*int(ncols)], nTrees, predictions, i*e.nClasses)
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
			for startIndex := range tasks {
				endIndex := startIndex + BatchSize
				if endIndex > nRows {
					endIndex = nRows
				}
				for i := startIndex; i < endIndex; i++ {
					e.predictInner(vals[i*int(ncols):(i+1)*int(ncols)], nTrees, predictions, i*e.nClasses)
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
