package util

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

func FindInBitsetUint32(bits uint32, pos uint32) bool {
	if pos >= 32 {
		return false
	}
	return (bits>>pos)&1 > 0
}

func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type stringParams map[string]string

func ReadParamsUntilBlank(reader *bufio.Reader) (stringParams, error) {
	params := make(stringParams)
	var line string
	var err error
	// skip empty
	for {
		line, err = reader.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimSpace(line)
		if line != "\n" {
			break
		}
	}
	for {
		tokens := strings.Split(line, "=")
		if len(tokens) > 2 {
			return nil, fmt.Errorf("meet wrong format while reading params")
		} else if len(tokens) == 2 {
			params[tokens[0]] = tokens[1]
		} else if len(tokens) == 1 {
			params[tokens[0]] = ""
		}
		line, err = reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
	}
	return params, nil
}

func (p *stringParams) ToInt(key string) (int, error) {
	valueStr, isFound := (*p)[key]
	if !isFound {
		return 0, fmt.Errorf("no %s field", key)
	}

	value, err := strconv.Atoi(valueStr)
	if err != nil {
		return 0, fmt.Errorf("can't convert %s: %s", key, err.Error())
	}
	return value, nil
}

func (p *stringParams) ToString(key string) (string, error) {
	valueStr, isFound := (*p)[key]
	if !isFound {
		return "", fmt.Errorf("no %s field", key)
	}
	return valueStr, nil
}

func (p *stringParams) Compare(key string, rhs string) error {
	valueStr, isFound := (*p)[key]
	if !isFound {
		return fmt.Errorf("no %s field", key)
	}

	if valueStr != rhs {
		return fmt.Errorf("only %s=%s is supported", key, rhs)
	}
	return nil
}

func (p *stringParams) ToStrSlice(key string) ([]string, error) {
	valueStr, isFound := (*p)[key]
	if !isFound {
		return nil, fmt.Errorf("no %s field", key)
	}
	return strings.Split(valueStr, " "), nil
}

func (p *stringParams) ToFloat64Slice(key string) ([]float64, error) {
	valueStr, isFound := (*p)[key]
	if !isFound {
		return nil, fmt.Errorf("no %s field", key)
	}
	valuesStr := strings.Split(valueStr, " ")
	values := make([]float64, len(valuesStr))
	for i, vStr := range valuesStr {
		value, err := strconv.ParseFloat(vStr, 64)
		if err != nil {
			return nil, fmt.Errorf("can't convert %s: %s", key, err.Error())
		}
		values[i] = value
	}
	return values, nil
}

func (p *stringParams) ToUint32Slice(key string) ([]uint32, error) {
	valueStr, isFound := (*p)[key]
	if !isFound {
		return nil, fmt.Errorf("no %s field", key)
	}
	valuesStr := strings.Split(valueStr, " ")
	values := make([]uint32, 0, len(valuesStr))
	for _, vStr := range valuesStr {
		value, err := strconv.ParseUint(vStr, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("can't convert %s: %s", key, err.Error())
		}
		values = append(values, uint32(value))
	}
	return values, nil
}

func (p *stringParams) ToInt32Slice(key string) ([]int32, error) {
	valueStr, isFound := (*p)[key]
	if !isFound {
		return nil, fmt.Errorf("no %s field", key)
	}
	valuesStr := strings.Split(valueStr, " ")
	values := make([]int32, 0, len(valuesStr))
	for _, vStr := range valuesStr {
		value, err := strconv.ParseInt(vStr, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("can't convert %s: %s", key, err.Error())
		}
		values = append(values, int32(value))
	}
	return values, nil
}

func (p *stringParams) Contains(key string) bool {
	_, isFound := (*p)[key]
	return isFound
}

var multiplyDeBruijnBitPosition = [...]uint32{
	0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
	31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9,
}

// https://stackoverflow.com/questions/757059/position-of-least-significant-bit-that-is-set
func FirstNonZeroBit(bitset []uint32) (uint32, error) {
	pos := uint32(0)
	for _, bitsetElement := range bitset {
		if bitsetElement > 0 {
			return pos + multiplyDeBruijnBitPosition[((bitsetElement&-bitsetElement)*0x077CB531)>>27], nil
		}
		pos += 32
	}
	return 0, fmt.Errorf("no bits set")
}

// https://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
func NumberOfSetBits(bitset []uint32) uint32 {
	numberOfSetBitsInBitsetElement := func(e uint32) uint32 {
		e = e - ((e >> 1) & 0x55555555)
		e = (e & 0x33333333) + ((e >> 2) & 0x33333333)
		return (((e + (e >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
	}
	count := uint32(0)
	for _, e := range bitset {
		count += numberOfSetBitsInBitsetElement(e)
	}
	return count
}

// ConstructBitset return a slice where bits in place of `values` are set
func ConstructBitset(values []int) []uint32 {
	if len(values) == 0 {
		return nil
	}
	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	max++

	nBitsetValues := max / 32
	if max%32 != 0 {
		nBitsetValues++
	}

	bitset := make([]uint32, nBitsetValues)
	for _, v := range values {
		i1 := v / 32
		i2 := v % 32
		bitset[i1] |= 1 << uint32(i2)
	}
	return bitset
}

func AlmostEqualFloat64(a, b, threshold float64) bool {
	return math.Abs(a-b) <= threshold
}

func AlmostEqualFloat64Slices(a, b []float64, threshold float64) error {
	if len(a) != len(b) {
		return fmt.Errorf("different sizes: len(a) = %d, len(b) = %d", len(a), len(b))
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > threshold {
			return fmt.Errorf("%d element mismatch: a[%d] = %f, b[%d] = %f", i, i, a[i], i, b[i])
		}
	}
	return nil
}

func NumMismatchedFloat64Slices(a, b []float64, threshold float64) (int, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("different sizes: len(a) = %d, len(b) = %d", len(a), len(b))
	}
	count := 0
	for i := range a {
		if math.Abs(a[i]-b[i]) > threshold {
			count++
		}
	}
	return count, nil
}

// Sigmoid applies sigmoid transformation to value
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidFloat64SliceInplace applies sigmoid transformation to each value in slice. Inplace
func SigmoidFloat64SliceInplace(vec []float64) {
	for i := range vec {
		vec[i] = Sigmoid(vec[i])
	}
}

func SoftmaxFloat64Slice(rawValues []float64, outputValues []float64, startIndex int) {
	sum := 0.0
	for i, v := range rawValues {
		exp := math.Exp(v)
		outputValues[startIndex+i] = exp
		sum += exp
	}
	if sum != 0.0 {
		inv_sum := 1.0 / sum
		for i := startIndex; i < startIndex+len(rawValues); i++ {
			outputValues[i] *= inv_sum
		}
	}
}

// Float64FromBytes converts 8 bytes to float
func Float64FromBytes(bytes []byte, littleEndian bool) float64 {
	var bits uint64
	if littleEndian {
		bits = binary.LittleEndian.Uint64(bytes)
	} else {
		bits = binary.BigEndian.Uint64(bytes)
	}
	return math.Float64frombits(bits)
}
