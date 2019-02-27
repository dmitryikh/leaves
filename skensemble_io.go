package leaves

import (
	"bufio"
	"fmt"
	"os"

	"github.com/dmitryikh/leaves/internal/pickle"
	"github.com/dmitryikh/leaves/transformation"
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

	if numNodes == 0 {
		// special case
		// we mimic decision rule but left and right childs lead to the same result
		t.nodes = make([]lgNode, 0, 1)
		node := numericalNode(0, 0, 0.0, 0)
		node.Flags |= leftLeaf
		node.Flags |= rightLeaf
		node.Left = uint32(len(t.leafValues))
		node.Right = uint32(len(t.leafValues))
		t.nodes = append(t.nodes, node)
		t.leafValues = append(t.leafValues, tree.Tree.Values[0]*scale+base)
		return t, nil
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
func SKEnsembleFromReader(reader *bufio.Reader, loadTransformation bool) (*Ensemble, error) {
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

	e.nRawOutputGroups = gbdt.NClasses
	if e.nRawOutputGroups == 2 {
		e.nRawOutputGroups = 1
	}

	e.MaxFeatureIdx = gbdt.MaxFeatures - 1

	nTrees := gbdt.NEstimators
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file")
	}

	if gbdt.NEstimators*e.nRawOutputGroups != len(gbdt.Estimators) {
		return nil, fmt.Errorf("unexpected number of trees (NEstimators = %d, nRawOutputGroups = %d, len(Estimatoers) = %d", gbdt.NEstimators, e.nRawOutputGroups, len(gbdt.Estimators))
	}

	scale := gbdt.LearningRate
	base := make([]float64, e.nRawOutputGroups)
	if gbdt.InitEstimator.Name == "LogOddsEstimator" {
		for i := 0; i < e.nRawOutputGroups; i++ {
			base[i] = gbdt.InitEstimator.Prior[0]
		}
	} else if gbdt.InitEstimator.Name == "PriorProbabilityEstimator" {
		if len(gbdt.InitEstimator.Prior) != len(base) {
			return nil, fmt.Errorf("len(gbdt.InitEstimator.Prior) != len(base)")
		}
		base = gbdt.InitEstimator.Prior
	} else {
		return nil, fmt.Errorf("unknown initial estimator \"%s\"", gbdt.InitEstimator.Name)
	}

	e.Trees = make([]lgTree, 0, gbdt.NEstimators*gbdt.NClasses)
	for i := 0; i < gbdt.NEstimators; i++ {
		for j := 0; j < e.nRawOutputGroups; j++ {
			treeNum := i*e.nRawOutputGroups + j
			tree, err := lgTreeFromSklearnDecisionTreeRegressor(gbdt.Estimators[treeNum], scale, base[j])
			if err != nil {
				return nil, fmt.Errorf("error while creating %d tree: %s", treeNum, err.Error())
			}
			e.Trees = append(e.Trees, tree)
		}
		for k := range base {
			base[k] = 0.0
		}
	}
	return &Ensemble{e, &transformation.TransformRaw{e.nRawOutputGroups}}, nil
}

// SKEnsembleFromFile reads sklearn tree ensemble model from pickle file
func SKEnsembleFromFile(filename string, loadTransformation bool) (*Ensemble, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return SKEnsembleFromReader(bufReader, loadTransformation)
}
