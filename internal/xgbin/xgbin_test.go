package xgbin

import (
	"bufio"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestReadGBTree(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "xgagaricus_previous_version.model")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)
	modelHeader, err := ReadModelHeader(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	trueModelHeader := &ModelHeader{}
	trueModelHeader.Param.NumFeatures = 127
	trueModelHeader.NameObj = "binary:logistic"
	trueModelHeader.NameGbm = "gbtree"
	if !reflect.DeepEqual(trueModelHeader, modelHeader) {
		t.Fatalf("unexpected ModelHeader values (got %v)", modelHeader)
	}

	gBTreeModel, err := ReadGBTreeModel(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	trueGBTreeModelParam := GBTreeModelParam{}
	trueGBTreeModelParam.NumTrees = 3
	trueGBTreeModelParam.DeprecatedNumRoots = 1
	trueGBTreeModelParam.DeprecatedNumFeature = 127
	trueGBTreeModelParam.DeprecatedNumOutputGroup = 1
	if !reflect.DeepEqual(trueGBTreeModelParam, gBTreeModel.Param) {
		t.Fatalf("unexpected GBTreeModelParam values (got %v)", gBTreeModel.Param)
	}

	trueTreeParamSlice := [3]TreeParam{}
	trueTreeParamSlice[0].NumRoots = 1
	trueTreeParamSlice[0].NumNodes = 7
	trueTreeParamSlice[0].NumFeature = 127
	trueTreeParamSlice[1].NumRoots = 1
	trueTreeParamSlice[1].NumNodes = 5
	trueTreeParamSlice[1].NumFeature = 127
	trueTreeParamSlice[2].NumRoots = 1
	trueTreeParamSlice[2].NumNodes = 7
	trueTreeParamSlice[2].NumFeature = 127
	if int32(len(gBTreeModel.Trees)) != gBTreeModel.Param.NumTrees {
		t.Fatalf("unexpected len(gBTreeModel.Trees) (got %d", len(gBTreeModel.Trees))
	}
	for i, tree := range gBTreeModel.Trees {
		if !reflect.DeepEqual(trueTreeParamSlice[i], tree.Param) {
			t.Fatalf("unexpected TreeParam values (got %v)", tree.Param)
		}
	}
	// NOTE: below I don't check values of trees because of float values comparison complexity..
	// TODO: add checks
}
