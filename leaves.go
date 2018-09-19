package leaves

// BatchSize for parallel task
const BatchSize = 16

// Ensemble is common interface that every model in leaves should implement
type Ensemble interface {
	PredictDense(vals []float64, nrows uint32, ncols uint32, predictions []float64, nTrees int, nThreads int) error
	PredictCSR(indptr []uint32, cols []uint32, vals []float64, predictions []float64, nTrees int, nThreads int)
	Predict(fvals []float64, nTrees int) float64
}
