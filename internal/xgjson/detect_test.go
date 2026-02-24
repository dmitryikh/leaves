package xgjson_test

import (
	"bufio"
	"bytes"
	"os"
	"testing"

	"github.com/dmitryikh/leaves/internal/xgjson"
)

const testdataDir = "testdata/"

func TestLooksLikeJSON(t *testing.T) {
	t.Run("real JSON model", func(t *testing.T) {
		r := openBuf(t, testdataDir+"test_binary_logistic.json")
		if ok, err := xgjson.LooksLikeJSON(r); err != nil || !ok {
			t.Errorf("LooksLikeJSON(json) = (%v, %v), want (true, nil)", ok, err)
		}
	})

	t.Run("UBJ file is not JSON", func(t *testing.T) {
		r := openBuf(t, testdataDir+"test_binary_logistic.ubj")
		if ok, _ := xgjson.LooksLikeJSON(r); ok {
			t.Error("LooksLikeJSON(ubj) = true, want false")
		}
	})

	t.Run("empty reader", func(t *testing.T) {
		r := bufio.NewReader(bytes.NewReader(nil))
		if ok, _ := xgjson.LooksLikeJSON(r); ok {
			t.Error("LooksLikeJSON(empty) = true, want false")
		}
	})

	t.Run("random bytes", func(t *testing.T) {
		r := bufio.NewReader(bytes.NewReader([]byte{0x00, 0x01, 0x02, 0x03, 0xFF}))
		if ok, _ := xgjson.LooksLikeJSON(r); ok {
			t.Error("LooksLikeJSON(random bytes) = true, want false")
		}
	})

	t.Run("reader is not consumed", func(t *testing.T) {
		f, err := os.Open(testdataDir + "test_binary_logistic.json")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		r := bufio.NewReader(f)
		xgjson.LooksLikeJSON(r) //nolint:errcheck
		// Peek must not consume: a second call should return the same result.
		if ok, _ := xgjson.LooksLikeJSON(r); !ok {
			t.Error("second LooksLikeJSON call returned false — reader was consumed")
		}
	})
}

func TestLooksLikeUBJ(t *testing.T) {
	t.Run("real UBJ model", func(t *testing.T) {
		r := openBuf(t, testdataDir+"test_binary_logistic.ubj")
		if ok, err := xgjson.LooksLikeUBJ(r); err != nil || !ok {
			t.Errorf("LooksLikeUBJ(ubj) = (%v, %v), want (true, nil)", ok, err)
		}
	})

	t.Run("JSON file is not UBJ", func(t *testing.T) {
		r := openBuf(t, testdataDir+"test_binary_logistic.json")
		if ok, _ := xgjson.LooksLikeUBJ(r); ok {
			t.Error("LooksLikeUBJ(json) = true, want false")
		}
	})

	t.Run("empty reader", func(t *testing.T) {
		r := bufio.NewReader(bytes.NewReader(nil))
		if ok, _ := xgjson.LooksLikeUBJ(r); ok {
			t.Error("LooksLikeUBJ(empty) = true, want false")
		}
	})

	t.Run("random bytes", func(t *testing.T) {
		r := bufio.NewReader(bytes.NewReader([]byte{0x00, 0x01, 0x02, 0x03, 0xFF}))
		if ok, _ := xgjson.LooksLikeUBJ(r); ok {
			t.Error("LooksLikeUBJ(random bytes) = true, want false")
		}
	})

	t.Run("reader is not consumed", func(t *testing.T) {
		f, err := os.Open(testdataDir + "test_binary_logistic.ubj")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		r := bufio.NewReader(f)
		xgjson.LooksLikeUBJ(r) //nolint:errcheck
		// Peek must not consume: a second call should return the same result.
		if ok, _ := xgjson.LooksLikeUBJ(r); !ok {
			t.Error("second LooksLikeUBJ call returned false — reader was consumed")
		}
	})
}

func openBuf(t *testing.T, path string) *bufio.Reader {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	t.Cleanup(func() { f.Close() })
	return bufio.NewReader(f)
}
