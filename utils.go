package leaves

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

func findInBitsetUint32(bits uint32, pos uint32) bool {
	if pos >= 32 {
		return false
	}
	return (bits>>pos)&1 > 0
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func readParamsUntilBlank(reader *bufio.Reader) (map[string]string, error) {
	params := make(map[string]string)
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

func mapValueToInt(params map[string]string, key string) (int, error) {
	valueStr, isFound := params[key]
	if !isFound {
		return 0, fmt.Errorf("no %s field", key)
	}

	value, err := strconv.Atoi(valueStr)
	if err != nil {
		return 0, fmt.Errorf("can't convert %s: %s", key, err.Error())
	}
	return value, nil
}

func mapValueCompare(params map[string]string, key string, rhs string) error {
	valueStr, isFound := params[key]
	if !isFound {
		return fmt.Errorf("no %s field", key)
	}

	if valueStr != rhs {
		return fmt.Errorf("only %s=%s is supported", key, rhs)
	}
	return nil
}

func mapValueToStrSlice(params map[string]string, key string) ([]string, error) {
	valueStr, isFound := params[key]
	if !isFound {
		return nil, fmt.Errorf("no %s field", key)
	}
	return strings.Split(valueStr, " "), nil
}

func mapValueToFloat64Slice(params map[string]string, key string) ([]float64, error) {
	valueStr, isFound := params[key]
	if !isFound {
		return nil, fmt.Errorf("no %s field", key)
	}
	valuesStr := strings.Split(valueStr, " ")
	values := make([]float64, 0, len(valuesStr))
	for _, vStr := range valuesStr {
		value, err := strconv.ParseFloat(vStr, 64)
		if err != nil {
			return nil, fmt.Errorf("can't convert %s: %s", key, err.Error())
		}
		values = append(values, value)
	}
	return values, nil
}

func mapValueToUint32Slice(params map[string]string, key string) ([]uint32, error) {
	valueStr, isFound := params[key]
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

func mapValueToInt32Slice(params map[string]string, key string) ([]int32, error) {
	valueStr, isFound := params[key]
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

var multiplyDeBruijnBitPosition = [...]uint32{
	0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
	31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9,
}

// https://stackoverflow.com/questions/757059/position-of-least-significant-bit-that-is-set
func firstNonZeroBit(bitset []uint32) (uint32, error) {
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
func numberOfSetBits(bitset []uint32) uint32 {

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

func almostEqualFloat64(a, b, threshold float64) bool {
	return math.Abs(a-b) <= threshold
}

func almostEqualFloat64Slices(a, b []float64, threshold float64) error {
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
