package mat

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

type libsvmRecord struct {
	column int
	value  float64
}

type libsvmRowFunc func(records []libsvmRecord) error

// DenseMatFromLibsvm reads dense matrix from libsvm format from `reader`
// stream. If `limit` > 0, reads only first limit `rows`. First colums is label,
// and usually you should set `skipFirstColumn` = true
func DenseMatFromLibsvm(reader *bufio.Reader, limit int, skipFirstColumn bool) (*DenseMat, error) {
	mat := &DenseMat{}
	f := func(records []libsvmRecord) error {
		return recordsToDenseMat(mat, records)
	}
	err := readFromLibsvm(reader, limit, skipFirstColumn, f)
	if err != nil {
		return nil, fmt.Errorf("unable to parse libsmv format to dense matrix: %s", err.Error())
	}
	return mat, nil
}

// DenseMatFromLibsvmFile reads dense matrix from libsvm file `filename`.
// If `limit` > 0, reads only first limit `rows`. First colums is label,
// and usually you should set `skipFirstColumn` = true
func DenseMatFromLibsvmFile(filename string, limit int, skipFirstColumn bool) (*DenseMat, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("unable to open %s: %s", filename, err.Error())
	}
	defer reader.Close()
	mat, err := DenseMatFromLibsvm(bufio.NewReader(reader), limit, skipFirstColumn)
	return mat, err
}

// CSRMatFromLibsvm reads CSR (Compressed Sparse Row) matrix from libsvm format
// from `reader` stream. If `limit` > 0, reads only first limit `rows`. First
// colums is label, and usually you should set `skipFirstColumn` = true
func CSRMatFromLibsvm(reader *bufio.Reader, limit int, skipFirstColumn bool) (*CSRMat, error) {
	mat := &CSRMat{}
	mat.RowHeaders = append(mat.RowHeaders, 0)
	f := func(records []libsvmRecord) error {
		return recordsToCSRMat(mat, records)
	}
	err := readFromLibsvm(reader, limit, skipFirstColumn, f)
	if err != nil {
		return nil, fmt.Errorf("unable to parse libsmv format to sparse matrix: %s", err.Error())
	}
	return mat, nil
}

// CSRMatFromLibsvmFile reads CSR (Compressed Sparse Row) matrix from libsvm file `filename`.
// If `limit` > 0, reads only first limit `rows`. First
// colums is label, and usually you should set `skipFirstColumn` = true
func CSRMatFromLibsvmFile(filename string, limit int, skipFirstColumn bool) (*CSRMat, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("unable to open %s: %s", filename, err.Error())
	}
	defer reader.Close()
	return CSRMatFromLibsvm(bufio.NewReader(reader), limit, skipFirstColumn)
}

// DenseMatFromCsv reads dense matrix from csv format with `delimiter` from
// `reader` stream. If `limit` > 0, reads only first limit `rows`. First colums
// is label, and usually you should set `skipFirstColumn` = true. If value is
// absent `defValue` will be used instead
func DenseMatFromCsv(reader *bufio.Reader,
	limit int,
	skipFirstColumn bool,
	delimiter string,
	defValue float64,
) (*DenseMat, error) {

	mat := &DenseMat{}
	f := func(records []libsvmRecord) error {
		return recordsToDenseMat(mat, records)
	}
	err := readFromCsv(reader, limit, skipFirstColumn, delimiter, defValue, f)
	if err != nil {
		return nil, fmt.Errorf("unable to parse csv format to dense matrix: %s", err.Error())
	}
	return mat, nil
}

// DenseMatFromCsvFile reads dense matrix from csv file `filename` with `delimiter`.
// If `limit` > 0, reads only first limit `rows`. First colums is label, and
// usually you should set `skipFirstColumn` = true. If value is absent
// `defValue` will be used instead
func DenseMatFromCsvFile(filename string,
	limit int,
	skipFirstColumn bool,
	delimiter string,
	defValue float64,
) (*DenseMat, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("unable to open %s: %s", filename, err.Error())
	}
	defer reader.Close()
	return DenseMatFromCsv(bufio.NewReader(reader), limit, skipFirstColumn, delimiter, defValue)
}

func recordsToDenseMat(mat *DenseMat, records []libsvmRecord) error {
	for i, r := range records {
		if i != r.column {
			return fmt.Errorf("wrong column number for dense matrix")
		}
		mat.Values = append(mat.Values, r.value)
	}
	if mat.Cols == 0 {
		mat.Cols = len(records)
	} else if mat.Cols != len(records) {
		return fmt.Errorf("different number of columns (%d != %d)", mat.Cols, len(records))
	}
	mat.Rows++
	return nil
}

func recordsToCSRMat(mat *CSRMat, records []libsvmRecord) error {
	for _, r := range records {
		mat.Values = append(mat.Values, r.value)
		mat.ColIndexes = append(mat.ColIndexes, r.column)
	}
	mat.RowHeaders = append(mat.RowHeaders, len(mat.Values))
	return nil
}

func readFromLibsvm(reader *bufio.Reader, limit int, skipFirstColumn bool, f libsvmRowFunc) error {
	records := make([]libsvmRecord, 0)
	startIndex := 0
	if skipFirstColumn {
		startIndex = 1
	}
	rows := 0
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		tokens := strings.Split(line, " ")
		if len(tokens) < 2 {
			return fmt.Errorf("too few columns")
		}

		records = records[:0]
		for col := startIndex; col < len(tokens); col++ {
			if len(tokens[col]) == 0 {
				break
			}
			pair := strings.Split(tokens[col], ":")
			if len(pair) != 2 {
				return fmt.Errorf("can't parse %s", tokens[col])
			}
			columnUint64, err := strconv.ParseUint(pair[0], 10, 32)
			column := int(columnUint64)
			if err != nil {
				return fmt.Errorf("can't convert to float %s: %s", pair[0], err.Error())
			}
			value, err := strconv.ParseFloat(pair[1], 64)
			if err != nil {
				return fmt.Errorf("can't convert to float %s: %s", pair[1], err.Error())
			}
			records = append(records, libsvmRecord{column, value})
		}

		err = f(records)
		if err != nil {
			return err
		}

		rows++
		if limit > 0 && rows == limit {
			break
		}
	}
	return nil
}

func readFromCsv(reader *bufio.Reader,
	limit int,
	skipFirstColumn bool,
	delimiter string,
	defValue float64,
	f libsvmRowFunc,
) error {
	records := make([]libsvmRecord, 0)
	rows := 0
	startIndex := 0
	if skipFirstColumn {
		startIndex = 1
	}
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		tokens := strings.Split(line, delimiter)

		records = records[:0]
		for col := startIndex; col < len(tokens); col++ {
			var value float64
			if len(tokens[col]) == 0 {
				value = defValue
			} else {
				fvalue, err := strconv.ParseFloat(tokens[col], 64)
				if err != nil {
					return fmt.Errorf("can't convert to float %s: %s", tokens[col], err.Error())
				}
				value = fvalue
			}
			records = append(records, libsvmRecord{col - startIndex, value})
		}
		err = f(records)
		if err != nil {
			return err
		}
		rows++
		if limit > 0 && rows == limit {
			break
		}
	}
	return nil
}

// WriteStr writes matrix to CSV like format with field delimiter `delimiter`
func (m *DenseMat) WriteStr(writer io.Writer, delimiter string) error {
	if m.Cols*m.Rows != len(m.Values) {
		return fmt.Errorf("matrix unconsistent")
	}
	if m.Cols == 0 || m.Rows == 0 {
		_, err := fmt.Fprint(writer, "\n")
		return err
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			_, err := fmt.Fprintf(writer, "%.19g", m.Values[i*m.Cols+j])
			if err != nil {
				return err
			}
			if j != m.Cols-1 {
				_, err := fmt.Fprint(writer, delimiter)
				if err != nil {
					return err
				}
			}
		}
		_, err := fmt.Fprint(writer, "\n")
		if err != nil {
			return err
		}
	}
	return nil
}

// ToCsvFile writes matrix to CSV like file
func (m *DenseMat) ToCsvFile(filename string, delimiter string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	buf := bufio.NewWriter(f)
	defer buf.Flush()
	return m.WriteStr(buf, delimiter)
}
