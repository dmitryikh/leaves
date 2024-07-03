package leaves

import (
	"context"
	"fmt"
	"golang.org/x/sync/errgroup"
	"math"
	"runtime"
	"sync"

	"github.com/manujz/leaves/transformation"
)

// BatchSize for parallel task
const BatchSize = 16

type ensembleBaseInterface interface {
	NEstimators() int
	NRawOutputGroups() int
	NFeatures() int
	NLeaves() []int
	Name() string
	adjustNEstimators(nEstimators int) int
	predictInner(ctx context.Context, fvals []float64, nEstimators int, predictions []float64, startIndex int) error
	predictLeafIndicesInner(ctx context.Context, fvals []float64, nEstimators int, predictions []float64, startIndex int) error
	resetFVals(fvals []float64)
}

// Ensemble is a common wrapper for all models
type Ensemble struct {
	ensembleBaseInterface
	transform transformation.Transform
}

func (e *Ensemble) predictInnerAndTransform(ctx context.Context, fvals []float64, nEstimators int, predictions []float64, startIndex int) error {
	if e.Transformation().Type() == transformation.Raw {
		return e.predictInner(ctx, fvals, nEstimators, predictions, startIndex)
	} else if e.Transformation().Type() == transformation.LeafIndex {
		return e.predictLeafIndicesInner(ctx, fvals, nEstimators, predictions, startIndex)
	} else {
		// TODO: avoid allocation here
		rawPredictions := make([]float64, e.NRawOutputGroups())
		if err := e.predictInner(ctx, fvals, nEstimators, rawPredictions, 0); err != nil {
			return err
		}
		return e.transform.Transform(rawPredictions, predictions, startIndex)
	}
}

func (e *Ensemble) checkNEstimators(nEstimators int) error {
	if e.transform.Type() == transformation.LeafIndex && nEstimators != e.NEstimators() {
		return fmt.Errorf("while predicting leaf indices all estimators should be used (provided num. of estimators = %d, should be %d)",
			nEstimators, e.NEstimators())
	}
	return nil
}

// PredictSingle calculates prediction for single class model. If ensemble is
// multiclass, will return quitely 0.0. Only `nEstimators` first estimators
// (trees in most cases) will be used. If `len(fvals)` is not enough function
// will quietly return 0.0.
// NOTE: for multiclass or leaf indices predictions use Predict
func (e *Ensemble) PredictSingle(ctx context.Context, fvals []float64, nEstimators int) (float64, error) {
	if e.NOutputGroups() != 1 {
		return 0.0, nil
	}
	if e.NFeatures() > len(fvals) {
		return 0.0, nil
	}
	nEstimators = e.adjustNEstimators(nEstimators)
	err := e.checkNEstimators(nEstimators)
	if err != nil {
		return 0.0, nil
	}
	ret := [1]float64{0.0}

	if err = e.predictInnerAndTransform(ctx, fvals, nEstimators, ret[:], 0); err != nil {
		return 0.0, err
	}
	return ret[0], nil
}

// Predict calculates single prediction for one or multiclass ensembles. Only
// `nEstimators` first estimators (trees in most cases) will be used.
// NOTE: for single class predictions one can use simplified function PredictSingle
func (e *Ensemble) Predict(ctx context.Context, fvals []float64, nEstimators int, predictions []float64) error {
	nRows := 1
	if len(predictions) < e.NOutputGroups()*nRows {
		return fmt.Errorf("predictions slice too short (should be at least %d)", e.NOutputGroups()*nRows)
	}
	if e.NFeatures() > len(fvals) {
		return fmt.Errorf("incorrect number of features (%d)", len(fvals))
	}
	nEstimators = e.adjustNEstimators(nEstimators)
	err := e.checkNEstimators(nEstimators)
	if err != nil {
		return err
	}

	return e.predictInnerAndTransform(ctx, fvals, nEstimators, predictions, 0)
}

// PredictCSR calculates predictions from ensemble. `indptr`, `cols`, `vals`
// represent data structures from Compressed Sparse Row Matrix format (see
// CSRMat). Only `nEstimators` first estimators (trees) will be used.
// `nThreads` points to number of threads that will be utilized (maximum
// is GO_MAX_PROCS)
// Note, `predictions` slice should be properly allocated on call side
func (e *Ensemble) PredictCSR(ctx context.Context, indptr []int, cols []int, vals []float64, predictions []float64, nEstimators int, nThreads int) error {
	nRows := len(indptr) - 1
	if len(predictions) < e.NOutputGroups()*nRows {
		return fmt.Errorf("predictions slice too short (should be at least %d)", e.NOutputGroups()*nRows)
	}
	nEstimators = e.adjustNEstimators(nEstimators)
	err := e.checkNEstimators(nEstimators)
	if err != nil {
		return err
	}

	if nRows <= BatchSize || nThreads == 0 || nThreads == 1 {
		// single thread calculations
		fvals := make([]float64, e.NFeatures())
		e.resetFVals(fvals)
		return e.predictCSRInner(ctx, indptr, cols, vals, 0, len(indptr)-1, predictions, nEstimators, fvals)
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

				e.predictCSRInner(ctx, indptr, cols, vals, startIndex, endIndex, predictions, nEstimators, fvals)
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
	ctx context.Context,
	indptr []int,
	cols []int,
	vals []float64,
	startIndex int,
	endIndex int,
	predictions []float64,
	nEstimators int,
	fvals []float64,
) error {
	for i := startIndex; i < endIndex; i++ {
		start := indptr[i]
		end := indptr[i+1]
		for j := start; j < end; j++ {
			if cols[j] < len(fvals) {
				fvals[cols[j]] = vals[j]
			}
		}

		if err := e.predictInnerAndTransform(ctx, fvals, nEstimators, predictions, i*e.NOutputGroups()); err != nil {
			return err
		}
		e.resetFVals(fvals)
	}
	return nil
}

// PredictDense calculates predictions from ensemble. `vals`, `rows`, `cols`
// represent data structures from Rom Major Matrix format (see DenseMat). Only
// `nEstimators` first estimators (trees in most cases) will be used. `nThreads`
// points to number of threads that will be utilized (maximum is GO_MAX_PROCS)
// Note, `predictions` slice should be properly allocated on call side
func (e *Ensemble) PredictDense(
	ctx context.Context,
	vals []float64,
	nrows int,
	ncols int,
	predictions []float64,
	nEstimators int,
	nThreads int,
) error {
	nRows := nrows
	if len(predictions) < e.NOutputGroups()*nRows {
		return fmt.Errorf("predictions slice too short (should be at least %d)", e.NOutputGroups()*nRows)
	}
	if ncols == 0 || e.NFeatures() > ncols {
		return fmt.Errorf("incorrect number of columns")
	}
	nEstimators = e.adjustNEstimators(nEstimators)
	err := e.checkNEstimators(nEstimators)
	if err != nil {
		return err
	}

	if nRows <= BatchSize || nThreads == 0 || nThreads == 1 {
		// single thread calculations
		for i := 0; i < nRows; i++ {
			fvals := vals[i*ncols : (i+1)*ncols]
			err = e.predictInnerAndTransform(ctx, fvals, nEstimators, predictions, i*e.NOutputGroups())
			if err != nil {
				return err
			}
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

	errgroup, ctx := errgroup.WithContext(ctx)
	for i := 0; i < nThreads; i++ {
		errgroup.Go(func() error {
			for startIndex := range tasks {
				endIndex := startIndex + BatchSize
				if endIndex > nRows {
					endIndex = nRows
				}
				for i := startIndex; i < endIndex; i++ {
					err = e.predictInnerAndTransform(ctx, vals[i*int(ncols):(i+1)*int(ncols)], nEstimators, predictions, i*e.NOutputGroups())
					if err != nil {
						return err
					}
				}
			}
			return nil
		})
	}

	// feed the queue
	for i := 0; i < nBatches; i++ {
		tasks <- i * BatchSize
	}
	close(tasks)
	return errgroup.Wait()
}

// NEstimators returns number of estimators (trees) in ensemble (per group)
func (e *Ensemble) NEstimators() int {
	return e.ensembleBaseInterface.NEstimators()
}

// NRawOutputGroups returns number of groups (numbers) in every object
// predictions before transformation function applied. This value is provided
// mainly for information purpose
func (e *Ensemble) NRawOutputGroups() int {
	return e.ensembleBaseInterface.NRawOutputGroups()
}

// NOutputGroups returns number of groups (numbers) in every object predictions.
// For example binary logistic model will give 1, but 4-class prediction model
// will give 4 numbers per object. This value usually used to preallocate slice
// for prediction values
func (e *Ensemble) NOutputGroups() int {
	return e.transform.NOutputGroups()
}

// NFeatures returns number of features in the model
func (e *Ensemble) NFeatures() int {
	return e.ensembleBaseInterface.NFeatures()
}

// NLeaves returns number of leaves in each tree of the ensemble. Returned
// vector has size NRawOutputGroups() * NEstimators(). For example to get number
// of leaves in group groupID for estimator estimatorID:
//
//	NLeaves()[groupID*NEstimators() + estimatorID].
//
// In case of NRawOutputGroups() == 1 (binary classification or regression):
//
//	NLeaves()[estimatorID]
func (e *Ensemble) NLeaves() []int {
	return e.ensembleBaseInterface.NLeaves()
}

// Name returns name of the estimator
func (e *Ensemble) Name() string {
	return e.ensembleBaseInterface.Name()
}

// Transformation returns transformation objects which applied to model outputs.
func (e *Ensemble) Transformation() transformation.Transform {
	return e.transform
}

// EnsembleWithRawPredictions returns ensemble instance with TransformRaw (no
// transformation functions will be applied to the model resulst)
func (e *Ensemble) EnsembleWithRawPredictions() *Ensemble {
	return &Ensemble{e, &transformation.TransformRaw{e.NRawOutputGroups()}}
}

// EnsembleWithLeafPredictions returns ensemble instance with TransformLeafIndex
// (return trees indices instead of numerical values)
func (e *Ensemble) EnsembleWithLeafPredictions() *Ensemble {
	// each predictions will produce NRawOutputGroups() * NEstimators() values
	return &Ensemble{e, &transformation.TransformLeafIndex{e.NRawOutputGroups() * e.NEstimators()}}
}
