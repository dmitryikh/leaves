package leaves

import (
	"bufio"
	"os"
	"path/filepath"
	"testing"
)

func TestLGMSLTR(t *testing.T) {
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
	reader, err = os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader = bufio.NewReader(reader)
	model, err := LGEnsembleFromReader(bufReader)
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
	model.PredictCSR(mat.RowHeaders, mat.ColIndexes, mat.Values, predictions, 0)

	// compare results
	if err := almostEqualFloat64Slices(truePredictions.Values, predictions, 1e-5); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func TestLGHiggs(t *testing.T) {
	// loading test data
	path := filepath.Join("testdata", "higgs_1000examples_test.libsvm")
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
	path = filepath.Join("testdata", "lghiggs.model")
	reader, err = os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader = bufio.NewReader(reader)
	model, err := LGEnsembleFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	path = filepath.Join("testdata", "lghiggs_1000examples_true_predictions.txt")
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
	model.PredictCSR(mat.RowHeaders, mat.ColIndexes, mat.Values, predictions, 0)

	// compare results
	if err := almostEqualFloat64Slices(truePredictions.Values, predictions, 1e-12); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func TestHiggs(t *testing.T) {
	// loading test data
	path := filepath.Join("testdata", "higgs_1000examples_test.libsvm")
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
	path = filepath.Join("testdata", "higgs.model")
	reader, err = os.Open(path)
	if err != nil {
		t.Skipf("Skipping due to absence of %s", path)
	}
	bufReader = bufio.NewReader(reader)
	model, err := LGEnsembleFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	// loading true predictions as DenseMat
	path = filepath.Join("testdata", "higgs_1000examples_true_predictions.txt")
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
	model.PredictCSR(mat.RowHeaders, mat.ColIndexes, mat.Values, predictions, 0)

	// compare results
	if err := almostEqualFloat64Slices(truePredictions.Values, predictions, 1e-5); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}

func BenchmarkLGMSLTR(b *testing.B) {
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
	reader, err = os.Open(path)
	if err != nil {
		b.Skipf("Skipping due to absence of %s", path)
	}
	bufReader = bufio.NewReader(reader)
	model, err := LGEnsembleFromReader(bufReader)
	if err != nil {
		b.Fatal(err)
	}

	// do benchmark
	b.ResetTimer()
	predictions := make([]float64, mat.Rows())
	for i := 0; i < b.N; i++ {
		model.PredictCSR(mat.RowHeaders, mat.ColIndexes, mat.Values, predictions, 0)
	}
}

func BenchmarkLGHiggs(b *testing.B) {
	// loading test data
	path := filepath.Join("testdata", "higgs_1000examples_test.libsvm")
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
	path = filepath.Join("testdata", "lghiggs.model")
	reader, err = os.Open(path)
	if err != nil {
		b.Skipf("Skipping due to absence of %s", path)
	}
	bufReader = bufio.NewReader(reader)
	model, err := LGEnsembleFromReader(bufReader)
	if err != nil {
		b.Fatal(err)
	}

	// do benchmark
	b.ResetTimer()
	predictions := make([]float64, mat.Rows())
	for i := 0; i < b.N; i++ {
		model.PredictCSR(mat.RowHeaders, mat.ColIndexes, mat.Values, predictions, 0)
	}
}

func TestXGAgaricus(t *testing.T) {
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
	reader, err = os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader = bufio.NewReader(reader)

	model, err := XGEnsembleFromReader(bufReader)
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
	model.PredictCSR(mat.RowHeaders, mat.ColIndexes, mat.Values, predictions, 0)
	SigmoidFloat64SliceInplace(predictions)
	// compare results
	if err := almostEqualFloat64Slices(truePredictions.Values, predictions, 1e-7); err != nil {
		t.Fatalf("different predictions: %s", err.Error())
	}
}
