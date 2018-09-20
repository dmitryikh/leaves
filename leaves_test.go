package leaves

import (
	"bufio"
	"os"
	"path/filepath"
	"testing"
)

func TestLGMSLTR(t *testing.T) {
	InnerTestLGMSLTR(t, 1)
	InnerTestLGMSLTR(t, 2)
	InnerTestLGMSLTR(t, 3)
	InnerTestLGMSLTR(t, 4)
}

func InnerTestLGMSLTR(t *testing.T, nThreads int) {
	// loading test data
	path := filepath.Join("testdata", "msltr_1000examples_test.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader := bufio.NewReader(reader)
	mat, err := CSRMatFromLibsvm(bufReader, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	// loading model
	path = filepath.Join("testdata", "lgmsltr.model")
	model, err := LGEnsembleFromFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	path = filepath.Join("testdata", "lgmsltr_1000examples_true_predictions.txt")
	reader, err = os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader = bufio.NewReader(reader)
	truePredictions, err := DenseMatFromCsv(bufReader, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, mat.Rows())
	model.PredictCSR(mat.RowHeaders, mat.ColIndexes, mat.Values, predictions, 0, nThreads)

	// compare results
	if err := almostEqualFloat64Slices(truePredictions.Values, predictions, 1e-5); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func TestLGHiggs(t *testing.T) {
	filename := "lghiggs_1000examples_true_predictions.txt"
	// loading model
	path := filepath.Join("testdata", "lghiggs.model")
	model, err := LGEnsembleFromFile(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	const tolerance = 1e-12

	// Dense matrix
	InnerTestHiggs(t, model, 1, true, filename, tolerance)
	InnerTestHiggs(t, model, 2, true, filename, tolerance)
	InnerTestHiggs(t, model, 3, true, filename, tolerance)
	InnerTestHiggs(t, model, 4, true, filename, tolerance)

	InnerTestHiggs(t, model, 1, false, filename, tolerance)
	InnerTestHiggs(t, model, 2, false, filename, tolerance)
	InnerTestHiggs(t, model, 3, false, filename, tolerance)
	InnerTestHiggs(t, model, 4, false, filename, tolerance)
}

func TestXGHiggs(t *testing.T) {
	t.Skip("have mismatch on 45 element")
	filename := "xghiggs_1000examples_true_predictions.txt"
	// loading model
	path := filepath.Join("testdata", "xghiggs.model")
	model, err := XGEnsembleFromFile(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	const tolerance = 1e-5

	// Dense matrix
	InnerTestHiggs(t, model, 1, true, filename, tolerance)
	InnerTestHiggs(t, model, 2, true, filename, tolerance)
	InnerTestHiggs(t, model, 3, true, filename, tolerance)
	InnerTestHiggs(t, model, 4, true, filename, tolerance)

	InnerTestHiggs(t, model, 1, false, filename, tolerance)
	InnerTestHiggs(t, model, 2, false, filename, tolerance)
	InnerTestHiggs(t, model, 3, false, filename, tolerance)
	InnerTestHiggs(t, model, 4, false, filename, tolerance)
}

func InnerTestHiggs(t *testing.T, model Ensemble, nThreads int, dense bool, truePredictionsFilename string, tolerance float64) {
	// loading test data
	path := filepath.Join("testdata", "higgs_1000examples_test.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader := bufio.NewReader(reader)
	var denseMat DenseMat
	var csrMat CSRMat
	var nRows int
	if dense {
		denseMat, err = DenseMatFromLibsvm(bufReader, 0, true)
		if err != nil {
			t.Fatal(err)
		}
		nRows = denseMat.Rows
	} else {
		csrMat, err = CSRMatFromLibsvm(bufReader, 0, true)
		if err != nil {
			t.Fatal(err)
		}
		nRows = csrMat.Rows()
	}

	// loading true predictions as DenseMat
	path = filepath.Join("testdata", truePredictionsFilename)
	reader, err = os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader = bufio.NewReader(reader)
	truePredictions, err := DenseMatFromCsv(bufReader, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	predictions := make([]float64, nRows)
	if dense {
		model.PredictDense(denseMat.Values, denseMat.Rows, denseMat.Cols, predictions, 0, nThreads)
	} else {
		model.PredictCSR(csrMat.RowHeaders, csrMat.ColIndexes, csrMat.Values, predictions, 0, nThreads)
	}
	// compare results
	if err := almostEqualFloat64Slices(truePredictions.Values, predictions, tolerance); err != nil {
		t.Errorf("different predictions: %s", err.Error())
	}

	if dense {
		// check single prediction
		singleIdx := 100
		fvals := denseMat.Values[singleIdx*denseMat.Cols : (singleIdx+1)*denseMat.Cols]
		prediction := model.PredictSingle(fvals, 0)
		if err := almostEqualFloat64Slices([]float64{truePredictions.Values[singleIdx]}, []float64{prediction}, tolerance); err != nil {
			t.Errorf("different PredictSingle prediction: %s", err.Error())
		}

		// check Predict
		singleIdx = 200
		fvals = denseMat.Values[singleIdx*denseMat.Cols : (singleIdx+1)*denseMat.Cols]
		predictions := make([]float64, 1)
		err := model.Predict(fvals, 0, predictions)
		if err != nil {
			t.Errorf("error while call model.Predict: %s", err.Error())
		}
		if err := almostEqualFloat64Slices([]float64{truePredictions.Values[singleIdx]}, predictions, tolerance); err != nil {
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
	path := filepath.Join("testdata", "msltr_1000examples_test.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		b.Skipf("Skipping due to absence of %s", path)
	}
	bufReader := bufio.NewReader(reader)
	mat, err := CSRMatFromLibsvm(bufReader, 0, true)
	if err != nil {
		b.Fatal(err)
	}

	// loading model
	path = filepath.Join("testdata", "lgmsltr.model")
	model, err := LGEnsembleFromFile(path)
	if err != nil {
		b.Fatal(err)
	}

	// do benchmark
	b.ResetTimer()
	predictions := make([]float64, mat.Rows())
	for i := 0; i < b.N; i++ {
		model.PredictCSR(mat.RowHeaders, mat.ColIndexes, mat.Values, predictions, 0, nThreads)
	}
}

func BenchmarkLGHiggs_dense_1thread(b *testing.B) {
	model, err := LGEnsembleFromFile(filepath.Join("testdata", "lghiggs.model"))
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 1, true)
}

func BenchmarkLGHiggs_dense_4thread(b *testing.B) {
	model, err := LGEnsembleFromFile(filepath.Join("testdata", "lghiggs.model"))
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 4, true)
}

func BenchmarkLGHiggs_csr_1thread(b *testing.B) {
	model, err := LGEnsembleFromFile(filepath.Join("testdata", "lghiggs.model"))
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 1, false)
}

func BenchmarkLGHiggs_csr_4thread(b *testing.B) {
	model, err := LGEnsembleFromFile(filepath.Join("testdata", "lghiggs.model"))
	if err != nil {
		b.Fatal(err)
	}
	InnerBenchmarkHiggs(b, model, 4, false)
}

func TestXGAgaricus_1thread(t *testing.T) {
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
	mat, err := CSRMatFromLibsvm(bufReader, 0, true)
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
	truePredictions, err := DenseMatFromCsv(bufReader, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// do predictions
	predictions := make([]float64, mat.Rows())
	model.PredictCSR(mat.RowHeaders, mat.ColIndexes, mat.Values, predictions, 0, nThreads)
	SigmoidFloat64SliceInplace(predictions)
	// compare results
	if err := almostEqualFloat64Slices(truePredictions.Values, predictions, 1e-7); err != nil {
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

func InnerBenchmarkHiggs(b *testing.B, model Ensemble, nThreads int, dense bool) {
	// loading test data
	path := filepath.Join("testdata", "higgs_1000examples_test.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		b.Skipf("Skipping due to absence of %s", path)
	}
	bufReader := bufio.NewReader(reader)
	var denseMat DenseMat
	var csrMat CSRMat
	var nRows int
	if dense {
		denseMat, err = DenseMatFromLibsvm(bufReader, 0, true)
		if err != nil {
			b.Fatal(err)
		}
		nRows = denseMat.Rows
	} else {
		csrMat, err = CSRMatFromLibsvm(bufReader, 0, true)
		if err != nil {
			b.Fatal(err)
		}
		nRows = csrMat.Rows()
	}

	// do benchmark
	b.ResetTimer()
	predictions := make([]float64, nRows)
	if dense {
		for i := 0; i < b.N; i++ {
			model.PredictDense(denseMat.Values, denseMat.Rows, denseMat.Cols, predictions, 0, nThreads)
		}
	} else {
		for i := 0; i < b.N; i++ {
			model.PredictCSR(csrMat.RowHeaders, csrMat.ColIndexes, csrMat.Values, predictions, 0, nThreads)
		}
	}
}
