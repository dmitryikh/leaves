package pickle

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestNdArrayPickle(t *testing.T) {
	path := filepath.Join("testdata", "ndarray.pickle0")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatalf("no file %s", path)
	}
	decoder := NewDecoder(reader)
	res, err := decoder.Decode()
	if err != nil {
		t.Fatalf("error while decoding: %s", err.Error())
	}
	array := NumpyArrayRaw{}
	err = ParseClass(&array, res)
	if err != nil {
		t.Fatalf("error while converting to numpy array: %s", err.Error())
	}
	if array.Type.LittleEndinan != true {
		t.Fatalf("expecting little endian")
	}
	if array.Type.Type != "i8" {
		t.Fatalf("expecting \"i8\"")
	}
	var parsedArray []int
	array.Data.Iterate(8, func(bytes []byte) error {
		parsedArray = append(parsedArray, int(binary.LittleEndian.Uint64(bytes)))
		return nil
	})

	trueArray := []int{1, 2, 3, 14, -10}
	if !reflect.DeepEqual(parsedArray, trueArray) {
		t.Fatalf("arrays mismatch")
	}
}

func TestDecisionTreeRegressorPickle(t *testing.T) {
	// see testdata/gradient_boosting_classifier.py for data generation
	path := filepath.Join("testdata", "decision_tree_regressor.pickle0")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatalf("no file %s", path)
	}
	decoder := NewDecoder(reader)
	res, err := decoder.Decode()
	if err != nil {
		t.Fatalf("error while decoding: %s", err.Error())
	}
	tree := SklearnDecisionTreeRegressor{}
	err = ParseClass(&tree, res)
	if err != nil {
		t.Fatalf("error while converting to SklearnDecisionTreeRegressor: %s", err.Error())
	}
	trueNodes := []SklearnNode{
		SklearnNode{LeftChild: 1, RightChild: 8, Feature: 18, Threshold: 0.19950735569000244, Impurity: 0.24999972299171574, NNodeSamples: 9500, WeightedNNodeSamples: 9500},
		SklearnNode{LeftChild: 2, RightChild: 5, Feature: 18, Threshold: -0.39447504281997681, Impurity: 0.08709949919090848, NNodeSamples: 4710, WeightedNNodeSamples: 4710},
		SklearnNode{LeftChild: 3, RightChild: 4, Feature: 18, Threshold: -0.77832573652267456, Impurity: 0.035496743113484275, NNodeSamples: 3663, WeightedNNodeSamples: 3663},
		SklearnNode{LeftChild: -1, RightChild: -1, Feature: -2, Threshold: -2, Impurity: 0.017814558539167552, NNodeSamples: 2866, WeightedNNodeSamples: 2866},
		SklearnNode{LeftChild: -1, RightChild: -1, Feature: -2, Threshold: -2, Impurity: 0.093295277617397893, NNodeSamples: 797, WeightedNNodeSamples: 797},
		SklearnNode{LeftChild: 6, RightChild: 7, Feature: 15, Threshold: 0.12563431262969971, Impurity: 0.21185011252422839, NNodeSamples: 1047, WeightedNNodeSamples: 1047},
		SklearnNode{LeftChild: -1, RightChild: -1, Feature: -2, Threshold: -2, Impurity: 0.12488796933696991, NNodeSamples: 458, WeightedNNodeSamples: 458},
		SklearnNode{LeftChild: -1, RightChild: -1, Feature: -2, Threshold: -2, Impurity: 0.24479348324257894, NNodeSamples: 589, WeightedNNodeSamples: 589},
		SklearnNode{LeftChild: 9, RightChild: 12, Feature: 18, Threshold: 0.57795190811157227, Impurity: 0.093322858599872827, NNodeSamples: 4790, WeightedNNodeSamples: 4790},
		SklearnNode{LeftChild: 10, RightChild: 11, Feature: 18, Threshold: 0.32796880602836609, Impurity: 0.21389097330753307, NNodeSamples: 842, WeightedNNodeSamples: 842},
		SklearnNode{LeftChild: -1, RightChild: -1, Feature: -2, Threshold: -2, Impurity: 0.24522440346354704, NNodeSamples: 246, WeightedNNodeSamples: 246},
		SklearnNode{LeftChild: -1, RightChild: -1, Feature: -2, Threshold: -2, Impurity: 0.19243221026080295, NNodeSamples: 596, WeightedNNodeSamples: 596},
		SklearnNode{LeftChild: 13, RightChild: 14, Feature: 18, Threshold: 0.71999788284301758, Impurity: 0.056649564911176425, NNodeSamples: 3948, WeightedNNodeSamples: 3948},
		SklearnNode{LeftChild: -1, RightChild: -1, Feature: -2, Threshold: -2, Impurity: 0.13888888888888959, NNodeSamples: 444, WeightedNNodeSamples: 444},
		SklearnNode{LeftChild: -1, RightChild: -1, Feature: -2, Threshold: -2, Impurity: 0.044613071036845198, NNodeSamples: 3504, WeightedNNodeSamples: 3504},
	}
	if !reflect.DeepEqual(tree.Tree.Nodes, trueNodes) {
		t.Fatalf("nodes mismatch")
	}
}
