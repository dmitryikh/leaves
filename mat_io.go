package leaves

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// DenseMatFromLibsvm reads dense matrix from libsvm format from `reader`
// stream. If `limit` > 0, reads only first limit `rows`. First colums is label,
// and usually you should set `skipFirstColumn` = true
func DenseMatFromLibsvm(reader *bufio.Reader, limit int, skipFirstColumn bool) (DenseMat, error) {
	mat := DenseMat{}
	startIndex := 0
	if skipFirstColumn {
		startIndex = 1
	}
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return mat, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		tokens := strings.Split(line, " ")
		if len(tokens) < 2 {
			return mat, fmt.Errorf("too few columns")
		}

		var column int
		for col := startIndex; col < len(tokens); col++ {
			if len(tokens[col]) == 0 {
				break
			}
			pair := strings.Split(tokens[col], ":")
			if len(pair) != 2 {
				return mat, fmt.Errorf("can't parse %s", tokens[col])
			}
			columnUint64, err := strconv.ParseUint(pair[0], 10, 32)
			column = int(columnUint64)
			if err != nil {
				return mat, fmt.Errorf("can't convert to float %s: %s", pair[0], err.Error())
			}
			if column != col-startIndex {
				return mat, fmt.Errorf("wrong column number for dense matrix")
			}
			fvalue, err := strconv.ParseFloat(pair[1], 64)
			if err != nil {
				return mat, fmt.Errorf("can't convert to float %s: %s", pair[1], err.Error())
			}
			mat.Values = append(mat.Values, fvalue)
		}
		if mat.Cols == 0 {
			mat.Cols = column + 1
		} else if mat.Cols != column+1 {
			return mat, fmt.Errorf("different number of columns (%d != %d)", mat.Cols, column+1)
		}

		mat.Rows++
		if limit > 0 && mat.Rows == limit {
			break
		}
	}
	return mat, nil
}

// CSRMatFromLibsvm reads CSR (Compressed Sparse Row) matrix from libsvm format
// from `reader` stream. If `limit` > 0, reads only first limit `rows`. First
// colums is label, and usually you should set `skipFirstColumn` = true
func CSRMatFromLibsvm(reader *bufio.Reader, limit int, skipFirstColumn bool) (CSRMat, error) {
	mat := CSRMat{}
	startIndex := 0
	if skipFirstColumn {
		startIndex = 1
	}
	rows := 0
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return mat, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		tokens := strings.Split(line, " ")
		if len(tokens) < 2 {
			return mat, fmt.Errorf("too few columns")
		}

		mat.RowHeaders = append(mat.RowHeaders, len(mat.Values))
		var column int
		for col := startIndex; col < len(tokens); col++ {
			if len(tokens[col]) == 0 {
				break
			}
			pair := strings.Split(tokens[col], ":")
			if len(pair) != 2 {
				return mat, fmt.Errorf("can't parse %s", tokens[col])
			}
			columnUint64, err := strconv.ParseUint(pair[0], 10, 32)
			column = int(columnUint64)
			if err != nil {
				return mat, fmt.Errorf("can't convert to float %s: %s", pair[0], err.Error())
			}
			fvalue, err := strconv.ParseFloat(pair[1], 64)
			if err != nil {
				return mat, fmt.Errorf("can't convert to float %s: %s", pair[1], err.Error())
			}
			mat.Values = append(mat.Values, fvalue)
			mat.ColIndexes = append(mat.ColIndexes, column)
		}

		rows++
		if limit > 0 && rows == limit {
			break
		}
	}
	mat.RowHeaders = append(mat.RowHeaders, len(mat.Values))
	return mat, nil
}

// DenseMatFromCsv reads dense matrix from csv format with `delimiter` from
// `reader` stream. If `limit` > 0, reads only first limit `rows`. First colums
// is label, and usually you should set `skipFirstColumn` = true. If value is
// absent `defValue` will be used instead
func DenseMatFromCsv(reader *bufio.Reader,
	limit int,
	skipFirstColumn bool,
	delimiter string,
	defValue float64) (DenseMat, error) {

	mat := DenseMat{}
	startIndex := 0
	if skipFirstColumn {
		startIndex = 1
	}
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return mat, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		tokens := strings.Split(line, delimiter)

		var column int
		for col := startIndex; col < len(tokens); col++ {
			var value float64
			if len(tokens[col]) == 0 {
				value = defValue
			} else {
				fvalue, err := strconv.ParseFloat(tokens[col], 64)
				if err != nil {
					return mat, fmt.Errorf("can't convert to float %s: %s", tokens[col], err.Error())
				}
				value = fvalue
			}
			mat.Values = append(mat.Values, value)
			column++
		}
		if mat.Cols == 0 {
			mat.Cols = column
		} else if mat.Cols != column {
			return mat, fmt.Errorf("different number of columns (%d != %d)", mat.Cols, column)
		}

		mat.Rows++
		if limit > 0 && mat.Rows == limit {
			break
		}
	}
	return mat, nil
}
