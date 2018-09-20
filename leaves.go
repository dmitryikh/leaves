package leaves

// BatchSize for parallel task
const BatchSize = 16

// Ensemble is common interface that every model in leaves should implement
type Ensemble interface {
	PredictDense(vals []float64, nrows int, ncols int, predictions []float64, nTrees int, nThreads int) error
	PredictCSR(indptr []int, cols []int, vals []float64, predictions []float64, nTrees int, nThreads int) error
	PredictSingle(fvals []float64, nTrees int) float64
	Predict(fvals []float64, nTrees int, predictions []float64) error
}
