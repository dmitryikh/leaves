package util

import (
	"bufio"
	"os"
	"path/filepath"
	"testing"
)

func TestFirstNonZeroBit(t *testing.T) {
	const length = 10
	const size = 32
	bitset := make([]uint32, length)
	_, err := FirstNonZeroBit(bitset)
	if err == nil {
		t.Error("all zeros bitset should fail")
	}

	check := func(trueAnswer uint32) {
		pos, err := FirstNonZeroBit(bitset)
		if err != nil {
			t.Error(err.Error())
		}
		if pos != trueAnswer {
			t.Errorf("%d fail", trueAnswer)
		}
	}

	bitset[9] |= 1 << 31
	check(9*size + 31)

	bitset[3] |= 1 << 3
	check(3*size + 3)

	bitset[0] |= 1 << 7
	check(7)

	bitset[0] |= 1 << 0
	check(0)
}

func TestNumberOfSetBits(t *testing.T) {
	const length = 10
	bitset := make([]uint32, length)

	check := func(trueAnswer uint32) {
		if NumberOfSetBits(bitset) != trueAnswer {
			t.Errorf("%d fail", trueAnswer)
		}
	}

	bitset[9] |= 1 << 31
	check(1)

	bitset[3] |= 1 << 3
	check(2)

	bitset[0] |= 1 << 7
	check(3)

	bitset[0] |= 1 << 0
	check(4)
}

func TestReadParams(t *testing.T) {
	path := filepath.Join("..", "testdata", "model_simple.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	// Читаем заголовок файла
	params, err := ReadParamsUntilBlank(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	trueMap := map[string]string{
		"version":                "v2",
		"num_class":              "1",
		"num_tree_per_iteration": "1",
		"label_index":            "0",
		"max_feature_idx":        "1",
		"objective":              "binary sigmoid:1",
		"feature_names":          "X1 X2",
		"feature_infos":          "[0:999] 1:0:3:100:-1",
		"tree_sizes":             "358 365",
	}
	for key, val := range trueMap {
		if params[key] != val {
			t.Errorf("params[%s] != %s", key, val)
		}
	}

	// Читаем первое дерево
	params, err = ReadParamsUntilBlank(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	trueMap = map[string]string{
		"Tree":           "0",
		"num_leaves":     "3",
		"num_cat":        "1",
		"split_feature":  "1 0",
		"split_gain":     "138.409 13.4409",
		"threshold":      "0 340.50000000000006",
		"decision_type":  "9 2",
		"left_child":     "-1 -2",
		"right_child":    "1 -3",
		"leaf_value":     "0.56697267424823339 0.3584987837673016 0.41213915936587919",
		"leaf_count":     "200 341 459",
		"internal_value": "0 -0.392018",
		"internal_count": "1000 800",
		"cat_boundaries": "0 4",
		"cat_threshold":  "0 0 0 16",
		"shrinkage":      "1",
	}
	for key, val := range trueMap {
		if params[key] != val {
			t.Errorf("params[%s] != %s", key, val)
		}
	}

	// Читаем второe дерево
	params, err = ReadParamsUntilBlank(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	trueMap = map[string]string{
		"Tree":           "1",
		"num_leaves":     "3",
		"num_cat":        "1",
		"split_feature":  "1 0",
		"split_gain":     "118.043 10.5922",
		"threshold":      "0 340.50000000000006",
		"decision_type":  "9 2",
		"left_child":     "-1 -2",
		"right_child":    "1 -3",
		"leaf_value":     "0.12883103567558912 -0.063872842243335157 -0.016484332942214807",
		"leaf_count":     "200 341 459",
		"internal_value": "0 -0.349854",
		"internal_count": "1000 800",
		"cat_boundaries": "0 4",
		"cat_threshold":  "0 0 0 16",
		"shrinkage":      "0.1",
	}
	for key, val := range trueMap {
		if params[key] != val {
			t.Errorf("params[%s] != %s", key, val)
		}
	}
}

func TestConstructBitset(t *testing.T) {
	bitset := ConstructBitset([]int{0})

	check := func(trueAnswer []uint32) {
		if len(trueAnswer) != len(bitset) {
			t.Errorf("wrong length. expected %d, got %d", len(trueAnswer), len(bitset))
		}
		for i, v := range trueAnswer {
			if v != bitset[i] {
				t.Errorf("wrong %d-th value. expected %d, got %d", i, v, bitset[i])
			}
		}
	}

	check([]uint32{1})

	bitset = ConstructBitset([]int{33, 65, 105})
	check([]uint32{0, 2, 2, 512})

	bitset = ConstructBitset([]int{})
	check([]uint32{})
}

func TestSigmoidFloat64SliceInplace(t *testing.T) {
	vec := [...]float64 {-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0}
	vecTrue := [...]float64 {0.26894142, 0.37754067, 0.4378235, 0.5, 0.5621765, 0.62245933, 0.73105858}
	SigmoidFloat64SliceInplace(vec[:])
	err := AlmostEqualFloat64Slices(vec[:], vecTrue[:], 1e-8)
	if err != nil {
		t.Error(err.Error())
	}
}

func TestSoftmaxFloat64Slice(t *testing.T) {
	compare := func(vec []float64, vecTrue []float64) {
		res := make([]float64, len(vec))
		SoftmaxFloat64Slice(vec, res, 0)
		err := AlmostEqualFloat64Slices(res, vecTrue, 1e-8)
		if err != nil {
			t.Error(err.Error())
		}
	}

	compare(
		[]float64{0.25, 0.75},
		[]float64{0.37754067, 0.62245933},
	)

	compare(
		[]float64{0.0, 0.0},
		[]float64{0.5, 0.5},
	)

	compare(
		[]float64{1.0, 2.0, 3.0},
		[]float64{0.09003057, 0.24472847, 0.66524096},
	)

	compare(
		[]float64{10.0, 20.0, 30.0},
		[]float64{2.06106005e-09, 4.53978686e-05, 9.99954600e-01},
	)

	compare(
		[]float64{},
		[]float64{},
	)
}
