package leaves

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// BatchSize for parallel task
const BatchSize = 16

type ensembleBaseInterface interface {
	NEstimators() int
	NClasses() int
	NFeatures() int
	Name() string
	adjustNEstimators(nEstimators int) int
	predictInner(fvals []float64, nEstimators int, predictions []float64, startIndex int)
	resetFVals(fvals []float64)
}

// Ensemble is a common wrapper for all models
type Ensemble struct {
	ensembleBaseInterface
}

// PredictSingle calculates prediction for single class model. If ensemble is
// multiclass, will return quitely 0.0. Only `nEstimators` first estimators
// (trees in most cases) will be used. If `len(fvals)` is not enough function
// will quietly return 0.0. Note, that result is a raw score (before sigmoid
// function transformation and etc)
// NOTE: for multiclass prediction use Predict
func (e *Ensemble) PredictSingle(fvals []float64, nEstimators int) float64 {
	if e.NClasses() != 1 {
		return 0.0
	}
	if e.NFeatures() > len(fvals) {
		return 0.0
	}
	nEstimators = e.adjustNEstimators(nEstimators)
	ret := [1]float64{0.0}

	e.predictInner(fvals, nEstimators, ret[:], 0)
	return ret[0]
}

// Predict calculates single prediction for one or multiclass ensembles. Only
// `nEstimators` first estimators (trees in most cases) will be used. Note, that
// result is a raw score (before sigmoid function transformation and etc)
// NOTE: for single class predictions one can use simplified function PredictSingle
func (e *Ensemble) Predict(fvals []float64, nEstimators int, predictions []float64) error {
	nRows := 1
	if len(predictions) < e.NClasses()*nRows {
		return fmt.Errorf("predictions slice too short (should be at least %d)", e.NClasses()*nRows)
	}
	if e.NFeatures() > len(fvals) {
		return fmt.Errorf("incorrect number of features (%d)", len(fvals))
	}
	nEstimators = e.adjustNEstimators(nEstimators)

	e.predictInner(fvals, nEstimators, predictions, 0)
	return nil
}

// PredictCSR calculates predictions from ensemble. `indptr`, `cols`, `vals`
// represent data structures from Compressed Sparse Row Matrix format (see
// CSRMat). Only `nEstimators` first estimators will be used (trees in most
// cases). `nThreads` points to number of threads that will be utilized (maximum
// is GO_MAX_PROCS)
// Note, that result is a raw score (before sigmoid function transformation and etc).
// Note, `predictions` slice should be properly allocated on call side
func (e *Ensemble) PredictCSR(indptr []int, cols []int, vals []float64, predictions []float64, nEstimators int, nThreads int) error {
	nRows := len(indptr) - 1
	if len(predictions) < e.NClasses()*nRows {
		return fmt.Errorf("predictions slice too short (should be at least %d)", e.NClasses()*nRows)
	}
	nEstimators = e.adjustNEstimators(nEstimators)
	if nRows <= BatchSize || nThreads == 0 || nThreads == 1 {
		// single thread calculations
		fvals := make([]float64, e.NFeatures())
		e.resetFVals(fvals)
		e.predictCSRInner(indptr, cols, vals, 0, len(indptr)-1, predictions, nEstimators, fvals)
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
			fvals := make([]float64, e.NFeatures())
			e.resetFVals(fvals)
			for startIndex := range tasks {
				endIndex := startIndex + BatchSize
				if endIndex > nRows {
					endIndex = nRows
				}
				e.predictCSRInner(indptr, cols, vals, startIndex, endIndex, predictions, nEstimators, fvals)
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

func (e *Ensemble) predictCSRInner(
	indptr []int,
	cols []int,
	vals []float64,
	startIndex int,
	endIndex int,
	predictions []float64,
	nEstimators int,
	fvals []float64,
) {
	for i := startIndex; i < endIndex; i++ {
		start := indptr[i]
		end := indptr[i+1]
		for j := start; j < end; j++ {
			if cols[j] < len(fvals) {
				fvals[cols[j]] = vals[j]
			}
		}
		e.predictInner(fvals, nEstimators, predictions, i*e.NClasses())
		e.resetFVals(fvals)
	}
}

// PredictDense calculates predictions from ensemble. `vals`, `rows`, `cols`
// represent data structures from Rom Major Matrix format (see DenseMat). Only
// `nEstimators` first estimators (trees in most cases) will be used. `nThreads`
// points to number of threads that will be utilized (maximum is GO_MAX_PROCS)
// Note, that result is a raw score (before sigmoid function transformation and etc).
// Note, `predictions` slice should be properly allocated on call side
func (e *Ensemble) PredictDense(
	vals []float64,
	nrows int,
	ncols int,
	predictions []float64,
	nEstimators int,
	nThreads int,
) error {
	nRows := nrows
	if len(predictions) < e.NClasses()*nRows {
		return fmt.Errorf("predictions slice too short (should be at least %d)", e.NClasses()*nRows)
	}
	if ncols == 0 || e.NFeatures() > ncols {
		return fmt.Errorf("incorrect number of columns")
	}
	nEstimators = e.adjustNEstimators(nEstimators)
	if nRows <= BatchSize || nThreads == 0 || nThreads == 1 {
		// single thread calculations
		for i := 0; i < nRows; i++ {
			e.predictInner(vals[i*ncols:(i+1)*ncols], nEstimators, predictions, i*e.NClasses())
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
					e.predictInner(vals[i*int(ncols):(i+1)*int(ncols)], nEstimators, predictions, i*e.NClasses())
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

// NEstimators returns number of estimators (trees) in ensemble (per class)
func (e *Ensemble) NEstimators() int {
	return e.ensembleBaseInterface.NEstimators()
}

// NClasses returns number of classes to predict
func (e *Ensemble) NClasses() int {
	return e.ensembleBaseInterface.NClasses()
}

// NFeatures returns number of features in the model
func (e *Ensemble) NFeatures() int {
	return e.ensembleBaseInterface.NFeatures()
}

// Name returns name of the estimator
func (e *Ensemble) Name() string {
	return e.ensembleBaseInterface.Name()
}
