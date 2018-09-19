package leaves

import (
	"fmt"
)

// DenseMat is dense matrix data structure
type DenseMat struct {
	Values []float64
	Cols   int
	Rows   int
}

// DenseMatFromArray converts arrays of `values` to DenseMat using shape
// information `rows` and `cols`
func DenseMatFromArray(values []float64, rows int, cols int) (DenseMat, error) {
	mat := DenseMat{}
	if len(values) != cols*rows {
		return mat, fmt.Errorf("wrong dimensions")
	}
	mat.Values = append(mat.Values, values...)
	mat.Cols = cols
	mat.Rows = rows
	return mat, nil
}

// CSRMat is Compressed Sparse Row matrix data structure
type CSRMat struct {
	RowHeaders []int
	ColIndexes []int
	Values     []float64
}

// Rows returns number of rows in the matrix
func (m *CSRMat) Rows() int {
	if len(m.RowHeaders) == 0 {
		return 0
	}
	return len(m.RowHeaders) - 1
}

// CSRMatFromArray converts arrays of `values` to CSRMat using shape information
// `rows` and `cols`. See also DenseMatFromArray to store dense data in matrix
func CSRMatFromArray(values []float64, rows int, cols int) (CSRMat, error) {
	mat := CSRMat{}
	if len(values) != cols*rows {
		return mat, fmt.Errorf("wrong dimensions")
	}
	mat.Values = append(mat.Values, values...)
	mat.ColIndexes = make([]int, 0, len(values))
	mat.RowHeaders = make([]int, 0, rows+1)

	for i := 0; i < rows; i++ {
		mat.RowHeaders = append(mat.RowHeaders, len(mat.ColIndexes))
		for j := 0; j < cols; j++ {
			mat.ColIndexes = append(mat.ColIndexes, j)
		}
	}
	mat.RowHeaders = append(mat.RowHeaders, len(mat.ColIndexes))
	return mat, nil
}
