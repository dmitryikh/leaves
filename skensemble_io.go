package leaves

import (
	"bufio"
	"fmt"
	"os"

	"github.com/dmitryikh/leaves/internal/pickle"
)

func lgTreeFromSklearnDecisionTreeRegressor(tree pickle.SklearnDecisionTreeRegressor, scale float64, base float64) (lgTree, error) {
	t := lgTree{}
	// no support for categorical features in sklearn trees
	t.nCategorical = 0

	numLeaves := 0
	numNodes := 0
	for _, n := range tree.Tree.Nodes {
		if n.LeftChild < 0 {
			numLeaves++
		} else {
			numNodes++
		}
	}

	if numLeaves-1 != numNodes {
		return t, fmt.Errorf("unexpected number of leaves (%d) and nodes (%d)", numLeaves, numNodes)
	}

	// Numerical only
	createNode := func(idx int) (lgNode, error) {
		node := lgNode{}
		refNode := &tree.Tree.Nodes[idx]
		missingType := uint8(0)
		defaultType := uint8(0)
		node = numericalNode(uint32(refNode.Feature), missingType, refNode.Threshold, defaultType)
		if tree.Tree.Nodes[refNode.LeftChild].LeftChild < 0 {
			node.Flags |= leftLeaf
			node.Left = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, tree.Tree.Values[refNode.LeftChild]*scale+base)
		}
		if tree.Tree.Nodes[refNode.RightChild].LeftChild < 0 {
			node.Flags |= rightLeaf
			node.Right = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, tree.Tree.Values[refNode.RightChild]*scale+base)
		}
		return node, nil
	}

	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, tree.Tree.NNodes)
	t.nodes = make([]lgNode, 0, numNodes)
	node, err := createNode(0)
	if err != nil {
		return t, err
	}
	t.nodes = append(t.nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)
	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		if t.nodes[convIdx].Flags&rightLeaf == 0 {
			origIdx := tree.Tree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].RightChild
			if !visited[origIdx] {
				node, err := createNode(origIdx)
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
			origIdx := tree.Tree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].LeftChild
			if !visited[origIdx] {
				node, err := createNode(origIdx)
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

// SKEnsembleFromReader reads sklearn tree ensemble model from `reader`
func SKEnsembleFromReader(reader *bufio.Reader) (*Ensemble, error) {
	e := &lgEnsemble{name: "sklearn.ensemble.GradientBoostingClassifier"}
	decoder := pickle.NewDecoder(reader)
	res, err := decoder.Decode()
	if err != nil {
		return nil, fmt.Errorf("error while decoding: %s", err.Error())
	}
	gbdt := pickle.SklearnGradientBoosting{}
	err = pickle.ParseClass(&gbdt, res)
	if err != nil {
		return nil, fmt.Errorf("error while parsing gradient boosting class: %s", err.Error())
	}

	e.nClasses = gbdt.NClasses
	if e.nClasses == 2 {
		e.nClasses = 1
	}

	e.MaxFeatureIdx = gbdt.MaxFeatures - 1

	nTrees := gbdt.NEstimators
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file")
	}

	scale := gbdt.LearningRate
	base := float64(0.0)
	if gbdt.InitEstimator.Name == "LogOddsEstimator" {
		base = gbdt.InitEstimator.Prior
	}

	e.Trees = make([]lgTree, 0, nTrees)
	for i := 0; i < nTrees; i++ {
		tree, err := lgTreeFromSklearnDecisionTreeRegressor(gbdt.Estimators[i], scale, base)
		if err != nil {
			return nil, fmt.Errorf("error while creating %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
		base = 0.0
	}
	fmt.Printf("%#v\n", e.Trees[0].leafValues)
	return &Ensemble{e}, nil
}

// SKEnsembleFromFile reads sklearn tree ensemble model from pickle file
func SKEnsembleFromFile(filename string) (*Ensemble, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return SKEnsembleFromReader(bufReader)
}
