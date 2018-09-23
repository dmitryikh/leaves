package leaves

import (
	"bufio"
	"fmt"
	"os"

	"github.com/dmitryikh/leaves/internal/xgbin"
)

func xgSplitIndex(origNode *xgbin.Node) uint32 {
	return origNode.SIndex & ((1 << 31) - 1)
}

func xgDefaultLeft(origNode *xgbin.Node) bool {
	return (origNode.SIndex >> 31) != 0
}

func xgIsLeaf(origNode *xgbin.Node) bool {
	return origNode.CLeft == -1
}

func xgTreeFromTreeModel(origTree *xgbin.TreeModel, numFeatures uint32) (lgTree, error) {
	t := lgTree{}

	if origTree.Param.NumFeature > int32(numFeatures) {
		return t, fmt.Errorf(
			"tree number of features %d, but header number of features %d",
			origTree.Param.NumFeature,
			numFeatures,
		)
	}

	if origTree.Param.NumRoots != 1 {
		return t, fmt.Errorf("support only trees with 1 root (got %d)", origTree.Param.NumRoots)
	}

	if origTree.Param.NumNodes == 0 {
		return t, fmt.Errorf("tree with zero number of nodes")
	}
	numNodes := origTree.Param.NumNodes

	// XGBoost doesn't support categorical features
	t.nCategorical = 0

	if numNodes == 1 {
		// special case
		// we mimic decision rule but left and right childs lead to the same result
		t.nodes = make([]lgNode, 0, numNodes)
		node := numericalNode(0, 0, 0.0, 0)
		node.Flags |= leftLeaf
		node.Flags |= rightLeaf
		node.Left = uint32(len(t.leafValues))
		node.Right = uint32(len(t.leafValues))
		t.leafValues = append(t.leafValues, float64(origTree.Nodes[0].Info))
		return t, nil
	}

	createNode := func(origNode *xgbin.Node) (lgNode, error) {
		node := lgNode{}
		// count nan as missing value
		// NOTE: this differs with XGBosst realization: could be a problem
		missingType := uint8(missingNan)

		defaultType := uint8(0)
		if xgDefaultLeft(origNode) {
			defaultType = defaultLeft
		}
		node = numericalNode(xgSplitIndex(origNode), missingType, float64(origNode.Info), defaultType)

		if origNode.CLeft < 0 {
			return node, fmt.Errorf("logic error: got origNode.CLeft < 0")
		}
		if origNode.CRight < 0 {
			return node, fmt.Errorf("logic error: got origNode.CRight < 0")
		}
		if xgIsLeaf(&origTree.Nodes[origNode.CLeft]) {
			node.Flags |= leftLeaf
			node.Left = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, float64(origTree.Nodes[origNode.CLeft].Info))
		}
		if xgIsLeaf(&origTree.Nodes[origNode.CRight]) {
			node.Flags |= rightLeaf
			node.Right = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, float64(origTree.Nodes[origNode.CRight].Info))
		}
		return node, nil
	}

	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, numNodes)
	t.nodes = make([]lgNode, 0, numNodes)
	node, err := createNode(&origTree.Nodes[0])
	if err != nil {
		return t, err
	}
	t.nodes = append(t.nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)
	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		if t.nodes[convIdx].Flags&rightLeaf == 0 {
			origIdx := origTree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].CRight
			if !visited[origIdx] {
				node, err := createNode(&origTree.Nodes[origIdx])
				if err != nil {
					return t, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Right = uint32(convNewIdx)
				continue
			}
		}
		if t.nodes[convIdx].Flags&leftLeaf == 0 {
			origIdx := origTree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].CLeft
			if !visited[origIdx] {
				node, err := createNode(&origTree.Nodes[origIdx])
				if err != nil {
					return t, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Left = uint32(convNewIdx)
				continue
			}
		}
		origNodeIdxStack = origNodeIdxStack[:len(origNodeIdxStack)-1]
		convNodeIdxStack = convNodeIdxStack[:len(convNodeIdxStack)-1]
	}
	return t, nil
}

// XGEnsembleFromReader reads  XGBoost model from `reader`
func XGEnsembleFromReader(reader *bufio.Reader) (*XGEnsemble, error) {
	e := &XGEnsemble{}

	// reading header info
	header, err := xgbin.ReadModelHeader(reader)
	if err != nil {
		return nil, err
	}
	if header.NameGbm != "gbtree" {
		return nil, fmt.Errorf("only gbtree is supported (got %s)", header.NameGbm)
	}
	if header.Param.NumFeatures == 0 {
		return nil, fmt.Errorf("zero number of features")
	}
	e.MaxFeatureIdx = int(header.Param.NumFeatures) - 1
	e.BaseScore = float64(header.Param.BaseScore)

	// reading gbtree
	origModel, err := xgbin.ReadGBTreeModel(reader)
	if err != nil {
		return nil, err
	}
	if origModel.Param.NumFeature > int32(header.Param.NumFeatures) {
		return nil, fmt.Errorf(
			"gbtee number of features %d, but header number of features %d",
			origModel.Param.NumFeature,
			header.Param.NumFeatures,
		)
	}
	// TODO: belowe is not true (see Agaricus test). Why?
	// if header.Param.NumClass != origModel.Param.NumOutputGroup {
	// 	return nil, fmt.Errorf("header number of class and model number of class should be the same (%d != %d)",
	// 		header.Param.NumClass, origModel.Param.NumOutputGroup)
	// }
	e.nClasses = int(origModel.Param.NumOutputGroup)
	if origModel.Param.NumRoots != 1 {
		return nil, fmt.Errorf("support only trees with 1 root (got %d)", origModel.Param.NumRoots)
	}
	e.TreeInfo = make([]int, len(origModel.TreeInfo))
	for i, v := range origModel.TreeInfo {
		e.TreeInfo[i] = int(v)
	}

	nTrees := origModel.Param.NumTrees
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in model")
	}

	// reading particular trees
	e.Trees = make([]lgTree, 0, nTrees)
	for i := int32(0); i < nTrees; i++ {
		tree, err := xgTreeFromTreeModel(origModel.Trees[i], header.Param.NumFeatures)
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
	}
	return e, nil
}

// XGEnsembleFromFile reads XGBoost model from binary file
func XGEnsembleFromFile(filename string) (*XGEnsemble, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return XGEnsembleFromReader(bufReader)
}
