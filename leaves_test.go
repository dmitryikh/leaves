package leaves

import (
	"bufio"
	"os"
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/mat"
	"github.com/dmitryikh/leaves/util"
)

func isFileExists(filename string) bool {
	f, err := os.Open(filename)
	if err != nil {
		return false
	}
	f.Close()
	return true
}

func skipTestIfFileNotExist(t *testing.T, filenames ...string) {
	for _, filename := range filenames {
		if !isFileExists(filename) {
			t.Skipf("Skipping due to absence of  file: %s", filename)
		}
	}
}

func skipBenchmarkIfFileNotExist(t *testing.B, filenames ...string) {
	for _, filename := range filenames {
		if !isFileExists(filename) {
			t.Skipf("Skipping due to absence of  file: %s", filename)
		}
	}
}

func TestLGMSLTR(t *testing.T) {
	InnerTestLGMSLTR(t, 1)
	InnerTestLGMSLTR(t, 2)
	InnerTestLGMSLTR(t, 3)
	InnerTestLGMSLTR(t, 4)
}

func InnerTestLGMSLTR(t *testing.T, nThreads int) {
	// loading test data
	testPath := filepath.Join("testdata", "msltr_1000examples_test.libsvm")
	truePath := filepath.Join("testdata", "lgmsltr_1000examples_true_predictions.txt")
	modelPath := filepath.Join("testdata", "lgmsltr.model")
	skipTestIfFileNotExist(t, testPath, truePath, modelPath)

	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	model, err := LGEnsembleFromFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}

	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, csr.Rows())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)

	// compare results
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, 1e-5); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func TestLGHiggs(t *testing.T) {
	truePath := filepath.Join("testdata", "lghiggs_1000examples_true_predictions.txt")
	modelPath := filepath.Join("testdata", "lghiggs.model")
	skipTestIfFileNotExist(t, truePath, modelPath)

	model, err := LGEnsembleFromFile(modelPath)
	if err != nil {
		t.Fatalf("fail loading model %s: %s", modelPath, err.Error())
	}

	const tolerance = 1e-12

	// Dense matrix
	InnerTestHiggs(t, model, 1, true, truePath, tolerance)
	InnerTestHiggs(t, model, 2, true, truePath, tolerance)
	InnerTestHiggs(t, model, 3, true, truePath, tolerance)
	InnerTestHiggs(t, model, 4, true, truePath, tolerance)
	// Sparse matrix
	InnerTestHiggs(t, model, 1, false, truePath, tolerance)
	InnerTestHiggs(t, model, 2, false, truePath, tolerance)
	InnerTestHiggs(t, model, 3, false, truePath, tolerance)
	InnerTestHiggs(t, model, 4, false, truePath, tolerance)
}

func TestXGHiggs(t *testing.T) {
	truePath := filepath.Join("testdata", "xghiggs_1000examples_true_predictions.txt")
	modelPath := filepath.Join("testdata", "xghiggs.model")
	skipTestIfFileNotExist(t, truePath, modelPath)

	// loading model
	model, err := XGEnsembleFromFile(modelPath)
	if err != nil {
		t.Fatalf("fail loading model %s: %s", modelPath, err.Error())
	}

	const tolerance = 1e-5

	// Dense matrix
	InnerTestHiggs(t, model, 1, true, truePath, tolerance)
	InnerTestHiggs(t, model, 2, true, truePath, tolerance)
	InnerTestHiggs(t, model, 3, true, truePath, tolerance)
	InnerTestHiggs(t, model, 4, true, truePath, tolerance)
	// Sparse matrix
	InnerTestHiggs(t, model, 1, false, truePath, tolerance)
	InnerTestHiggs(t, model, 2, false, truePath, tolerance)
	InnerTestHiggs(t, model, 3, false, truePath, tolerance)
	InnerTestHiggs(t, model, 4, false, truePath, tolerance)
}

func InnerTestHiggs(t *testing.T, model Ensemble, nThreads int, isDense bool, truePredictionsFilename string, tolerance float64) {
	// loading test data
	testPath := filepath.Join("testdata", "higgs_1000examples_test.libsvm")
	skipTestIfFileNotExist(t, testPath)

	var dense *mat.DenseMat
	var csr *mat.CSRMat
	var nRows int
	var err error
	if isDense {
		dense, err = mat.DenseMatFromLibsvmFile(testPath, 0, true)
		if err != nil {
			t.Fatal(err)
		}
		nRows = dense.Rows
	} else {
		csr, err = mat.CSRMatFromLibsvmFile(testPath, 0, true)
		if err != nil {
			t.Fatal(err)
		}
		nRows = csr.Rows()
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePredictionsFilename, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	predictions := make([]float64, nRows)
	if isDense {
		model.PredictDense(dense.Values, dense.Rows, dense.Cols, predictions, 0, nThreads)
	} else {
		model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
	}
	// compare results. Count number of mismatched values beacase of floating point
	// comparisons problems: fval < thresholds.
	// I think this is because float32 format inside of XGBoost Binary format
	count, err := util.NumMismatchedFloat64Slices(truePredictions.Values, predictions, tolerance)
	if err != nil {
		t.Errorf(err.Error())
	}

	if count > 70 {
		t.Errorf("mismatched more than %d predictions", count)
	}

	if isDense {
		// check single prediction
		singleIdx := 100
		fvals := dense.Values[singleIdx*dense.Cols : (singleIdx+1)*dense.Cols]
		prediction := model.PredictSingle(fvals, 0)
		if err := util.AlmostEqualFloat64Slices([]float64{truePredictions.Values[singleIdx]}, []float64{prediction}, tolerance); err != nil {
			t.Errorf("different PredictSingle prediction: %s", err.Error())
		}

		// check Predict
		singleIdx = 200
		fvals = dense.Values[singleIdx*dense.Cols : (singleIdx+1)*dense.Cols]
		predictions := make([]float64, 1)
		err := model.Predict(fvals, 0, predictions)
		if err != nil {
			t.Errorf("error while call model.Predict: %s", err.Error())
		}
		if err := util.AlmostEqualFloat64Slices([]float64{truePredictions.Values[singleIdx]}, predictions, tolerance); err != nil {
			t.Errorf("different Predict prediction: %s", err.Error())
		}
	}
}

func BenchmarkLGMSLTR_csr_1thread(b *testing.B) {
	InnerBenchmarkLGMSLTR(b, 1)
}

func BenchmarkLGMSLTR_csr_4thread(b *testing.B) {
	InnerBenchmarkLGMSLTR(b, 4)
}

func InnerBenchmarkLGMSLTR(b *testing.B, nThreads int) {
	// loading test data
	testPath := filepath.Join("testdata", "msltr_1000examples_test.libsvm")
	modelPath := filepath.Join("testdata", "lgmsltr.model")
	skipBenchmarkIfFileNotExist(b, testPath, modelPath)
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		b.Fatal(err)
	}

	// loading model
	model, err := LGEnsembleFromFile(modelPath)
	if err != nil {
		b.Fatal(err)
	}

	// do benchmark
	b.ResetTimer()
	predictions := make([]float64, csr.Rows())
	for i := 0; i < b.N; i++ {
		model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
	}
}

func BenchmarkLGHiggs_dense_1thread(b *testing.B) {
	modelPath := filepath.Join("testdata", "lghiggs.model")
	skipBenchmarkIfFileNotExist(b, modelPath)
	model, err := LGEnsembleFromFile(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 1, true)
}

func BenchmarkLGHiggs_dense_4thread(b *testing.B) {
	modelPath := filepath.Join("testdata", "lghiggs.model")
	skipBenchmarkIfFileNotExist(b, modelPath)
	model, err := LGEnsembleFromFile(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 4, true)
}

func BenchmarkLGHiggs_csr_1thread(b *testing.B) {
	modelPath := filepath.Join("testdata", "lghiggs.model")
	skipBenchmarkIfFileNotExist(b, modelPath)
	model, err := LGEnsembleFromFile(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 1, false)
}

func BenchmarkLGHiggs_csr_4thread(b *testing.B) {
	modelPath := filepath.Join("testdata", "lghiggs.model")
	skipBenchmarkIfFileNotExist(b, modelPath)
	model, err := LGEnsembleFromFile(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 4, false)
}

func TestXGAgaricus(t *testing.T) {
	InnerTestXGAgaricus(t, 1)
	InnerTestXGAgaricus(t, 2)
	InnerTestXGAgaricus(t, 3)
	InnerTestXGAgaricus(t, 4)
}

func InnerTestXGAgaricus(t *testing.T, nThreads int) {
	// loading test data
	path := filepath.Join("testdata", "agaricus_test.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader := bufio.NewReader(reader)
	csr, err := mat.CSRMatFromLibsvm(bufReader, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	path = filepath.Join("testdata", "xgagaricus.model")
	model, err := XGEnsembleFromFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if model.NTrees() != 3 {
		t.Fatalf("expected 3 trees (got %d)", model.NTrees())
	}

	// loading true predictions as DenseMat
	path = filepath.Join("testdata", "xgagaricus_true_predictions.txt")
	reader, err = os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader = bufio.NewReader(reader)
	truePredictions, err := mat.DenseMatFromCsv(bufReader, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, csr.Rows())
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
	util.SigmoidFloat64SliceInplace(predictions)
	// compare results
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, 1e-7); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func BenchmarkXGHiggs_dense_1thread(b *testing.B) {
	model, err := XGEnsembleFromFile(filepath.Join("testdata", "xghiggs.model"))
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 1, true)
}

func BenchmarkXGHiggs_dense_4thread(b *testing.B) {
	model, err := XGEnsembleFromFile(filepath.Join("testdata", "xghiggs.model"))
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 4, true)
}

func BenchmarkXGHiggs_csr_1thread(b *testing.B) {
	model, err := XGEnsembleFromFile(filepath.Join("testdata", "xghiggs.model"))
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 1, false)
}

func BenchmarkXGHiggs_csr_4thread(b *testing.B) {
	model, err := XGEnsembleFromFile(filepath.Join("testdata", "xghiggs.model"))
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 4, false)
}

func InnerBenchmarkHiggs(b *testing.B, model Ensemble, nThreads int, isDense bool) {
	// loading test data
	truePath := filepath.Join("testdata", "higgs_1000examples_test.libsvm")
	skipBenchmarkIfFileNotExist(b, truePath)
	var dense *mat.DenseMat
	var csr *mat.CSRMat
	var nRows int
	var err error
	if isDense {
		dense, err = mat.DenseMatFromLibsvmFile(truePath, 0, true)
		if err != nil {
			b.Fatal(err)
		}
		nRows = dense.Rows
	} else {
		csr, err = mat.CSRMatFromLibsvmFile(truePath, 0, true)
		if err != nil {
			b.Fatal(err)
		}
		nRows = csr.Rows()
	}

	// do benchmark
	b.ResetTimer()
	predictions := make([]float64, nRows)
	if isDense {
		for i := 0; i < b.N; i++ {
			model.PredictDense(dense.Values, dense.Rows, dense.Cols, predictions, 0, nThreads)
		}
	} else {
		for i := 0; i < b.N; i++ {
			model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
		}
	}
}

func TestLGMulticlass(t *testing.T) {
	InnerTestLGMulticlass(t, 1)
	InnerTestLGMulticlass(t, 2)
	InnerTestLGMulticlass(t, 3)
	InnerTestLGMulticlass(t, 4)
}

func InnerTestLGMulticlass(t *testing.T, nThreads int) {
	// loading test data
	testPath := filepath.Join("testdata", "multiclass_test.tsv")
	modelPath := filepath.Join("testdata", "lgmulticlass.model")
	truePath := filepath.Join("testdata", "lgmulticlass_true_predictions.txt")
	skipTestIfFileNotExist(t, testPath, modelPath, truePath)
	dense, err := mat.DenseMatFromCsvFile(testPath, 0, true, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := LGEnsembleFromFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if model.NTrees() != 10 {
		t.Fatalf("expected 10 trees (got %d)", model.NTrees())
	}
	if model.NClasses() != 5 {
		t.Fatalf("expected 5 classes (got %d)", model.NClasses())
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, dense.Rows*model.nClasses)
	model.PredictDense(dense.Values, dense.Rows, dense.Cols, predictions, 0, nThreads)
	// compare results
	const tolerance = 1e-7
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}

	// check Predict
	singleIdx := 200
	fvals := dense.Values[singleIdx*dense.Cols : (singleIdx+1)*dense.Cols]
	predictions = make([]float64, model.NClasses())
	err = model.Predict(fvals, 0, predictions)
	if err != nil {
		t.Errorf("error while call model.Predict: %s", err.Error())
	}
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values[singleIdx*model.NClasses():(singleIdx+1)*model.NClasses()], predictions, tolerance); err != nil {
		t.Errorf("different Predict prediction: %s", err.Error())
	}
}

func BenchmarkHiggsLoading(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := XGEnsembleFromFile(filepath.Join("testdata", "xghiggs.model"))
		if err != nil {
			b.Skip(err.Error())
		}
	}
}

func TestXGDermatology(t *testing.T) {
	InnerTestXGDermatology(t, 1)
	InnerTestXGDermatology(t, 2)
	InnerTestXGDermatology(t, 3)
	InnerTestXGDermatology(t, 4)
}

func InnerTestXGDermatology(t *testing.T, nThreads int) {
	// loading test data
	testPath := filepath.Join("testdata", "dermatology_test.libsvm")
	modelPath := filepath.Join("testdata", "xgdermatology.model")
	truePath := filepath.Join("testdata", "xgdermatology_true_predictions.txt")
	skipTestIfFileNotExist(t, testPath, modelPath, truePath)
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	model, err := XGEnsembleFromFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, "\t", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, csr.Rows()*model.nClasses)
	model.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, predictions, 0, nThreads)
	// compare results
	const tolerance = 1e-6
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}
}
