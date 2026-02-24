// Package ubjdecode implements a standalone Universal Binary JSON (UBJSON) decoder.
// Zero external dependencies — uses only encoding/binary and io from stdlib.
//
// UBJSON spec: https://ubjson.org/
// XGBoost uses UBJSON for its .ubj model format.
package ubjdecode

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// DecodeValue reads one UBJSON value from r and returns it as a native Go value:
//
//	map[string]interface{} for objects
//	[]interface{} for untyped/mixed arrays
//	[]int32, []int64, []float32, []float64 for optimized typed arrays
//	string, bool, int64, float64, nil for scalars
func DecodeValue(r io.Reader) (interface{}, error) {
	marker, err := readByte(r)
	if err != nil {
		return nil, err
	}
	return decodeMarker(r, marker)
}

func decodeMarker(r io.Reader, marker byte) (interface{}, error) {
	for marker == 'N' { // noop — skip and read next
		var err error
		marker, err = readByte(r)
		if err != nil {
			return nil, err
		}
	}

	switch marker {
	case 'Z': // null
		return nil, nil
	case 'T': // true
		return true, nil
	case 'F': // false
		return false, nil
	case 'i': // int8
		b, err := readByte(r)
		if err != nil {
			return nil, err
		}
		return int64(int8(b)), nil
	case 'U': // uint8
		b, err := readByte(r)
		if err != nil {
			return nil, err
		}
		return int64(b), nil
	case 'I': // int16 big-endian
		var v int16
		if err := binary.Read(r, binary.BigEndian, &v); err != nil {
			return nil, err
		}
		return int64(v), nil
	case 'l': // int32 big-endian
		var v int32
		if err := binary.Read(r, binary.BigEndian, &v); err != nil {
			return nil, err
		}
		return int64(v), nil
	case 'L': // int64 big-endian
		var v int64
		if err := binary.Read(r, binary.BigEndian, &v); err != nil {
			return nil, err
		}
		return v, nil
	case 'd': // float32 big-endian
		var bits uint32
		if err := binary.Read(r, binary.BigEndian, &bits); err != nil {
			return nil, err
		}
		return float64(math.Float32frombits(bits)), nil
	case 'D': // float64 big-endian
		var bits uint64
		if err := binary.Read(r, binary.BigEndian, &bits); err != nil {
			return nil, err
		}
		return math.Float64frombits(bits), nil
	case 'C': // char — single byte as string
		b, err := readByte(r)
		if err != nil {
			return nil, err
		}
		return string([]byte{b}), nil
	case 'S': // string — length marker + bytes
		n, err := readLength(r)
		if err != nil {
			return nil, fmt.Errorf("string length: %w", err)
		}
		buf := make([]byte, n)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		return string(buf), nil
	case 'H': // high-precision number — decode as string
		n, err := readLength(r)
		if err != nil {
			return nil, fmt.Errorf("high-precision length: %w", err)
		}
		buf := make([]byte, n)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		return string(buf), nil
	case '[': // array
		return decodeArray(r)
	case '{': // object
		return decodeObject(r)
	default:
		return nil, fmt.Errorf("ubjdecode: unknown marker byte 0x%02X ('%c')", marker, marker)
	}
}

// readByte reads a single byte from r.
func readByte(r io.Reader) (byte, error) {
	var buf [1]byte
	_, err := io.ReadFull(r, buf[:])
	return buf[0], err
}

// readLength reads an integer length value (used for string lengths, array/object counts).
// The next byte is a type marker ('i','U','I','l','L') indicating the integer width.
func readLength(r io.Reader) (int64, error) {
	marker, err := readByte(r)
	if err != nil {
		return 0, err
	}
	switch marker {
	case 'i':
		b, err := readByte(r)
		return int64(int8(b)), err
	case 'U':
		b, err := readByte(r)
		return int64(b), err
	case 'I':
		var v int16
		err := binary.Read(r, binary.BigEndian, &v)
		return int64(v), err
	case 'l':
		var v int32
		err := binary.Read(r, binary.BigEndian, &v)
		return int64(v), err
	case 'L':
		var v int64
		err := binary.Read(r, binary.BigEndian, &v)
		return v, err
	default:
		return 0, fmt.Errorf("ubjdecode: unexpected length marker 0x%02X ('%c')", marker, marker)
	}
}

// decodeArray decodes a UBJSON array. The '[' marker has already been consumed.
// Handles all four container forms:
//  1. [$][type][#][len] — typed + counted → returns typed Go slice
//  2. [#][len]          — counted only    → returns []interface{}
//  3. [$][type] + ']'   — typed, terminated
//  4. ']' terminator    — untyped, terminated → returns []interface{}
func decodeArray(r io.Reader) (interface{}, error) {
	next, err := readByte(r)
	if err != nil {
		return nil, err
	}

	// Typed array: [$][type][#][count]
	if next == '$' {
		typeMarker, err := readByte(r)
		if err != nil {
			return nil, err
		}
		// Peek for '#'
		after, err := readByte(r)
		if err != nil {
			return nil, err
		}
		if after == '#' {
			// typed + counted: fast path
			count, err := readLength(r)
			if err != nil {
				return nil, fmt.Errorf("typed array count: %w", err)
			}
			return readTypedArray(r, typeMarker, int(count))
		}
		// typed + unterminated: read until ']'
		// `after` is the first element's marker? No — in typed arrays the element
		// marker is NOT repeated; `after` here should be ']' or data bytes.
		// Actually when typed without count, each element has no marker prefix —
		// the type is fixed. So `after` is the first data byte of the first element.
		// We must decode accordingly.
		return readTypedArrayUntilEnd(r, typeMarker, after)
	}

	// Counted only: [#][len]
	if next == '#' {
		count, err := readLength(r)
		if err != nil {
			return nil, fmt.Errorf("counted array length: %w", err)
		}
		result := make([]interface{}, 0, int(count))
		for i := int64(0); i < count; i++ {
			v, err := DecodeValue(r)
			if err != nil {
				return nil, fmt.Errorf("array element %d: %w", i, err)
			}
			result = append(result, v)
		}
		return result, nil
	}

	// Unterminated untyped: read until ']'
	if next == ']' {
		return []interface{}{}, nil
	}
	// next is the marker of the first element
	result := []interface{}{}
	v, err := decodeMarker(r, next)
	if err != nil {
		return nil, fmt.Errorf("array element 0: %w", err)
	}
	result = append(result, v)
	for {
		marker, err := readByte(r)
		if err != nil {
			return nil, err
		}
		if marker == ']' {
			break
		}
		v, err := decodeMarker(r, marker)
		if err != nil {
			return nil, fmt.Errorf("array element: %w", err)
		}
		result = append(result, v)
	}
	return result, nil
}

// readTypedArray reads `count` elements of a fixed type without per-element markers.
// Returns typed Go slices for performance-critical XGBoost arrays.
func readTypedArray(r io.Reader, typeMarker byte, count int) (interface{}, error) {
	switch typeMarker {
	case 'i': // int8 → []int32
		buf := make([]int8, count)
		for i := range buf {
			b, err := readByte(r)
			if err != nil {
				return nil, err
			}
			buf[i] = int8(b)
		}
		out := make([]int32, count)
		for i, v := range buf {
			out[i] = int32(v)
		}
		return out, nil
	case 'U': // uint8 → []int32
		buf := make([]byte, count)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		out := make([]int32, count)
		for i, v := range buf {
			out[i] = int32(v)
		}
		return out, nil
	case 'I': // int16 → []int32
		out := make([]int32, count)
		for i := range out {
			var v int16
			if err := binary.Read(r, binary.BigEndian, &v); err != nil {
				return nil, err
			}
			out[i] = int32(v)
		}
		return out, nil
	case 'l': // int32 → []int32
		out := make([]int32, count)
		for i := range out {
			if err := binary.Read(r, binary.BigEndian, &out[i]); err != nil {
				return nil, err
			}
		}
		return out, nil
	case 'L': // int64 → []int64
		out := make([]int64, count)
		for i := range out {
			if err := binary.Read(r, binary.BigEndian, &out[i]); err != nil {
				return nil, err
			}
		}
		return out, nil
	case 'd': // float32 → []float32
		out := make([]float32, count)
		for i := range out {
			var bits uint32
			if err := binary.Read(r, binary.BigEndian, &bits); err != nil {
				return nil, err
			}
			out[i] = math.Float32frombits(bits)
		}
		return out, nil
	case 'D': // float64 → []float64
		out := make([]float64, count)
		for i := range out {
			var bits uint64
			if err := binary.Read(r, binary.BigEndian, &bits); err != nil {
				return nil, err
			}
			out[i] = math.Float64frombits(bits)
		}
		return out, nil
	default:
		// Fall back to []interface{} for other typed arrays
		result := make([]interface{}, 0, count)
		for i := 0; i < count; i++ {
			v, err := decodeMarker(r, typeMarker)
			if err != nil {
				return nil, fmt.Errorf("typed array element %d: %w", i, err)
			}
			result = append(result, v)
		}
		return result, nil
	}
}

// readTypedArrayUntilEnd reads a typed array without a count, terminated by ']'.
// firstDataByte is the first raw data byte already consumed (not a marker).
func readTypedArrayUntilEnd(r io.Reader, typeMarker byte, firstDataByte byte) (interface{}, error) {
	// For typed arrays without count, we need to reconstruct the first element
	// and then keep reading until we see ']'.
	// Strategy: build []interface{} since we don't know the count.
	result := []interface{}{}

	// Decode first element from firstDataByte
	first, err := decodeSingleFromByte(typeMarker, firstDataByte, r)
	if err != nil {
		return nil, err
	}
	result = append(result, first)

	for {
		// For typed arrays, next byte is raw data (no marker), or ']' to end
		b, err := readByte(r)
		if err != nil {
			return nil, err
		}
		if b == ']' {
			break
		}
		v, err := decodeSingleFromByte(typeMarker, b, r)
		if err != nil {
			return nil, err
		}
		result = append(result, v)
	}
	return result, nil
}

// decodeSingleFromByte decodes one element of a typed array given the first
// byte already consumed and any remaining bytes read from r.
func decodeSingleFromByte(typeMarker byte, firstByte byte, r io.Reader) (interface{}, error) {
	switch typeMarker {
	case 'i':
		return int64(int8(firstByte)), nil
	case 'U':
		return int64(firstByte), nil
	case 'I':
		var buf [1]byte
		if _, err := io.ReadFull(r, buf[:]); err != nil {
			return nil, err
		}
		v := int16(firstByte)<<8 | int16(buf[0])
		return int64(v), nil
	case 'l':
		var rest [3]byte
		if _, err := io.ReadFull(r, rest[:]); err != nil {
			return nil, err
		}
		v := int32(firstByte)<<24 | int32(rest[0])<<16 | int32(rest[1])<<8 | int32(rest[2])
		return int64(v), nil
	case 'L':
		var rest [7]byte
		if _, err := io.ReadFull(r, rest[:]); err != nil {
			return nil, err
		}
		v := int64(firstByte)<<56 | int64(rest[0])<<48 | int64(rest[1])<<40 |
			int64(rest[2])<<32 | int64(rest[3])<<24 | int64(rest[4])<<16 |
			int64(rest[5])<<8 | int64(rest[6])
		return v, nil
	case 'd':
		var rest [3]byte
		if _, err := io.ReadFull(r, rest[:]); err != nil {
			return nil, err
		}
		bits := uint32(firstByte)<<24 | uint32(rest[0])<<16 | uint32(rest[1])<<8 | uint32(rest[2])
		return float64(math.Float32frombits(bits)), nil
	case 'D':
		var rest [7]byte
		if _, err := io.ReadFull(r, rest[:]); err != nil {
			return nil, err
		}
		bits := uint64(firstByte)<<56 | uint64(rest[0])<<48 | uint64(rest[1])<<40 |
			uint64(rest[2])<<32 | uint64(rest[3])<<24 | uint64(rest[4])<<16 |
			uint64(rest[5])<<8 | uint64(rest[6])
		return math.Float64frombits(bits), nil
	default:
		// treat firstByte as the marker
		return decodeMarker(r, firstByte)
	}
}

// decodeObject decodes a UBJSON object. The '{' marker has already been consumed.
// Returns map[string]interface{}.
func decodeObject(r io.Reader) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// Check for optimized form: [{][#][len] or [{][$][type][#][len]
	next, err := readByte(r)
	if err != nil {
		return nil, err
	}

	if next == '#' {
		// counted object
		count, err := readLength(r)
		if err != nil {
			return nil, fmt.Errorf("object count: %w", err)
		}
		for i := int64(0); i < count; i++ {
			k, v, err := readKeyValue(r)
			if err != nil {
				return nil, fmt.Errorf("object key/value %d: %w", i, err)
			}
			result[k] = v
		}
		return result, nil
	}

	if next == '}' {
		return result, nil
	}

	// Unoptimized: next byte is the length marker of the first key
	// In UBJSON objects, keys are NOT prefixed with 'S' — just length_marker + bytes
	k, v, err := readKeyValueFromLenMarker(r, next)
	if err != nil {
		return nil, fmt.Errorf("object first key/value: %w", err)
	}
	result[k] = v

	for {
		marker, err := readByte(r)
		if err != nil {
			return nil, err
		}
		if marker == '}' {
			break
		}
		k, v, err := readKeyValueFromLenMarker(r, marker)
		if err != nil {
			return nil, fmt.Errorf("object key/value: %w", err)
		}
		result[k] = v
	}
	return result, nil
}

// readKeyValue reads one key-value pair from an object.
// The key length marker has NOT been consumed yet.
func readKeyValue(r io.Reader) (string, interface{}, error) {
	lenMarker, err := readByte(r)
	if err != nil {
		return "", nil, err
	}
	return readKeyValueFromLenMarker(r, lenMarker)
}

// readKeyValueFromLenMarker reads one key-value pair where lenMarker is the
// already-consumed first byte of the key length.
func readKeyValueFromLenMarker(r io.Reader, lenMarker byte) (string, interface{}, error) {
	// lenMarker is the type marker for the key length
	var keyLen int64
	switch lenMarker {
	case 'i':
		b, err := readByte(r)
		if err != nil {
			return "", nil, err
		}
		keyLen = int64(int8(b))
	case 'U':
		b, err := readByte(r)
		if err != nil {
			return "", nil, err
		}
		keyLen = int64(b)
	case 'I':
		var v int16
		if err := binary.Read(r, binary.BigEndian, &v); err != nil {
			return "", nil, err
		}
		keyLen = int64(v)
	case 'l':
		var v int32
		if err := binary.Read(r, binary.BigEndian, &v); err != nil {
			return "", nil, err
		}
		keyLen = int64(v)
	case 'L':
		var v int64
		if err := binary.Read(r, binary.BigEndian, &v); err != nil {
			return "", nil, err
		}
		keyLen = v
	default:
		return "", nil, fmt.Errorf("ubjdecode: unexpected key length marker 0x%02X ('%c')", lenMarker, lenMarker)
	}

	keyBuf := make([]byte, keyLen)
	if _, err := io.ReadFull(r, keyBuf); err != nil {
		return "", nil, fmt.Errorf("reading key: %w", err)
	}
	key := string(keyBuf)

	val, err := DecodeValue(r)
	if err != nil {
		return "", nil, fmt.Errorf("value for key %q: %w", key, err)
	}
	return key, val, nil
}
