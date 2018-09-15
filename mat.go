package leaves

import (
	"fmt"
)

type DenseMat struct {
	Values []float64
	Cols   uint32
	Rows   uint32
}

func DenseMatFromArray(values []float64, rows uint32, cols uint32) (DenseMat, error) {
	mat := DenseMat{}
	if uint32(len(values)) != cols*rows {
		return mat, fmt.Errorf("wrong dimensions")
	}
	mat.Values = append(mat.Values, values...)
	mat.Cols = cols
	mat.Rows = rows
	return mat, nil
}

type CSRMat struct {
	RowHeaders []uint32
	ColIndexes []uint32
	Values     []float64
}

func (m *CSRMat) Rows() uint32 {
	if len(m.RowHeaders) == 0 {
		return 0
	}
	return uint32(len(m.RowHeaders)) - 1
}

func CSRMatFromArray(values []float64, rows uint32, cols uint32) (CSRMat, error) {
	mat := CSRMat{}
	if uint32(len(values)) != cols*rows {
		return mat, fmt.Errorf("wrong dimensions")
	}
	mat.Values = append(mat.Values, values...)
	mat.ColIndexes = make([]uint32, 0, len(values))
	mat.RowHeaders = make([]uint32, 0, rows+1)

	for i := uint32(0); i < rows; i++ {
		mat.RowHeaders = append(mat.RowHeaders, uint32(len(mat.ColIndexes)))
		for j := uint32(0); j < cols; j++ {
			mat.ColIndexes = append(mat.ColIndexes, j)
		}
	}
	mat.RowHeaders = append(mat.RowHeaders, uint32(len(mat.ColIndexes)))
	return mat, nil
}
