package mat

import (
	"bufio"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/dmitryikh/leaves/util"
)

func TestDenseMatFromLibsvm(t *testing.T) {
	path := filepath.Join("..", "testdata", "densemat.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	_, err = DenseMatFromLibsvm(bufReader, 0, false)
	if err == nil {
		t.Fatal("should fail because of first column")
	}

	// check reading correctness
	reader.Seek(0, 0)
	bufReader = bufio.NewReader(reader)
	mat, err := DenseMatFromLibsvm(bufReader, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if mat.Cols != 3 {
		t.Errorf("mat.Cols should be 3 (got %d)", mat.Cols)
	}
	if mat.Rows != 2 {
		t.Errorf("mat.Rows should be 2 (got %d)", mat.Rows)
	}
	trueValues := []float64{19.0, 45.3, 1e-6, 14.0, 0.0, 0.0}
	if err := util.AlmostEqualFloat64Slices(mat.Values, trueValues, 1e-10); err != nil {
		t.Errorf("mat.Values incorrect: %s", err.Error())
	}

	// check reading correctness with limit 1
	reader.Seek(0, 0)
	bufReader = bufio.NewReader(reader)
	mat, err = DenseMatFromLibsvm(bufReader, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	if mat.Cols != 3 {
		t.Errorf("mat.Cols should be 3 (got %d)", mat.Cols)
	}
	if mat.Rows != 1 {
		t.Errorf("mat.Rows should be 1 (got %d)", mat.Rows)
	}
	trueValues = []float64{19.0, 45.3, 1e-6}
	if err := util.AlmostEqualFloat64Slices(mat.Values, trueValues, 1e-10); err != nil {
		t.Errorf("mat.Values incorrect: %s", err.Error())
	}
}

func TestCSRMatFromLibsvm(t *testing.T) {
	path := filepath.Join("..", "testdata", "csrmat.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	_, err = CSRMatFromLibsvm(bufReader, 0, false)
	if err == nil {
		t.Fatal("should fail because of first column")
	}

	// check reading correctness
	reader.Seek(0, 0)
	bufReader = bufio.NewReader(reader)
	mat, err := CSRMatFromLibsvm(bufReader, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	trueValues := []float64{19.0, 45.3, 1e-6, 14.0, 0.0}
	if err := util.AlmostEqualFloat64Slices(mat.Values, trueValues, 1e-10); err != nil {
		t.Errorf("mat.Values incorrect: %s", err.Error())
	}

	trueRowHeaders := []int{0, 3, 5}
	if !reflect.DeepEqual(mat.RowHeaders, trueRowHeaders) {
		t.Error("mat.RowHeaders are incorrect")
	}

	trueColIndexes := []int{0, 10, 12, 4, 5}
	if !reflect.DeepEqual(mat.ColIndexes, trueColIndexes) {
		t.Error("mat.ColIndexes are incorrect")
	}

	// check reading correctness with limit 1
	reader.Seek(0, 0)
	bufReader = bufio.NewReader(reader)
	mat, err = CSRMatFromLibsvm(bufReader, 1, true)
	if err != nil {
		t.Fatal(err)
	}

	trueValues = []float64{19.0, 45.3, 1e-6}
	if err := util.AlmostEqualFloat64Slices(mat.Values, trueValues, 1e-10); err != nil {
		t.Errorf("mat.Values incorrect: %s", err.Error())
	}

	trueRowHeaders = []int{0, 3}
	if !reflect.DeepEqual(mat.RowHeaders, trueRowHeaders) {
		t.Error("mat.RowHeaders are incorrect")
	}

	trueColIndexes = []int{0, 10, 12}
	if !reflect.DeepEqual(mat.ColIndexes, trueColIndexes) {
		t.Error("mat.ColIndexes are incorrect")
	}
}
