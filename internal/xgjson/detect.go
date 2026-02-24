package xgjson

import (
	"bufio"
	"encoding/binary"
	"bytes"
)

// LooksLikeJSON peeks at the buffered reader and returns true if the content
// appears to be an XGBoost JSON model file. It does not advance the reader.
//
// Heuristic: the first non-whitespace byte must be '{', and the string
// "\"learner\"" (the mandatory top-level key in every XGBoost model) must
// appear within the first 512 peeked bytes.
func LooksLikeJSON(r *bufio.Reader) (bool, error) {
	buf, err := r.Peek(512)
	if len(buf) == 0 {
		return false, err
	}

	// Find first non-whitespace byte; it must be '{'
	firstNonWS := -1
	for i, b := range buf {
		if b != ' ' && b != '\t' && b != '\r' && b != '\n' {
			firstNonWS = i
			break
		}
	}
	if firstNonWS < 0 || buf[firstNonWS] != '{' {
		return false, nil
	}

	// Confirm the mandatory top-level key is present in the peeked window.
	// Every XGBoost JSON model has "learner" as its first object key.
	return bytes.Contains(buf, []byte(`"learner"`)), nil
}

// LooksLikeUBJ peeks at the buffered reader and returns true if the content
// appears to be an XGBoost UBJ (Universal Binary JSON) model file. It does
// not advance the reader.
//
// Heuristic: UBJ objects start with '{' (0x7B), and the first key in an
// XGBoost UBJ model is always "learner". We verify the first byte is '{' and
// then parse the UBJ key-length encoding + key bytes from the peek buffer,
// checking that the first key is exactly "learner".
//
// UBJ key lengths are encoded as: <type-marker> <integer-bytes>, where the
// marker is one of: 'i' (int8), 'U' (uint8), 'I' (int16 BE), 'l' (int32 BE),
// 'L' (int64 BE). XGBoost currently uses 'L', so we handle all widths for
// robustness.
func LooksLikeUBJ(r *bufio.Reader) (bool, error) {
	// 32 bytes is more than enough: 1 ('{') + 1 (marker) + 8 (int64) + 7 ("learner") = 17
	buf, err := r.Peek(32)
	if len(buf) < 2 {
		return false, err
	}

	if buf[0] != '{' {
		return false, nil
	}

	// Parse the key length from buf[1:]
	keyLen, headerLen, ok := ubjReadLength(buf[1:])
	if !ok {
		return false, nil
	}

	// "learner" is 7 bytes
	if keyLen != 7 {
		return false, nil
	}

	start := 1 + headerLen
	end := start + 7
	if end > len(buf) {
		return false, nil
	}

	return bytes.Equal(buf[start:end], []byte("learner")), nil
}

// ubjReadLength parses a UBJ integer-length from b (starting at b[0] which is
// the type marker). Returns (value, bytesConsumed, ok).
func ubjReadLength(b []byte) (value int, consumed int, ok bool) {
	if len(b) < 1 {
		return 0, 0, false
	}
	marker := b[0]
	switch marker {
	case 'i': // int8 — 1 data byte
		if len(b) < 2 {
			return 0, 0, false
		}
		return int(int8(b[1])), 2, true
	case 'U': // uint8 — 1 data byte
		if len(b) < 2 {
			return 0, 0, false
		}
		return int(b[1]), 2, true
	case 'I': // int16 BE — 2 data bytes
		if len(b) < 3 {
			return 0, 0, false
		}
		v := int16(binary.BigEndian.Uint16(b[1:3]))
		return int(v), 3, true
	case 'l': // int32 BE — 4 data bytes
		if len(b) < 5 {
			return 0, 0, false
		}
		v := int32(binary.BigEndian.Uint32(b[1:5]))
		return int(v), 5, true
	case 'L': // int64 BE — 8 data bytes
		if len(b) < 9 {
			return 0, 0, false
		}
		v := binary.BigEndian.Uint64(b[1:9])
		return int(v), 9, true
	default:
		return 0, 0, false
	}
}
