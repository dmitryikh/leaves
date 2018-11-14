package leaves

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/dmitryikh/leaves/util"
)

func convertMissingType(decisionType uint32) (uint8, error) {
	missingTypeOrig := (decisionType >> 2) & 3
	missingType := uint8(0)
	if missingTypeOrig == 0 {
		// default value
	} else if missingTypeOrig == 1 {
		missingType = missingZero
	} else if missingTypeOrig == 2 {
		missingType = missingNan
	} else {
		return 0, fmt.Errorf("unknown missing type = %d", missingTypeOrig)
	}
	return missingType, nil
}

func lgTreeFromReader(reader *bufio.Reader) (lgTree, error) {
	t := lgTree{}
	params, err := util.ReadParamsUntilBlank(reader)
	if err != nil {
		return t, err
	}
	numCategorical, err := params.ToInt("num_cat")
	if err != nil {
		return t, err
	}
	t.nCategorical = uint32(numCategorical)

	numLeaves, err := params.ToInt("num_leaves")
	if err != nil {
		return t, err
	}
	if numLeaves < 1 {
		return t, fmt.Errorf("num_leaves < 1")
	}
	numNodes := numLeaves - 1

	leafValues, err := params.ToFloat64Slice("leaf_value")
	if err != nil {
		return t, err
	}
	t.leafValues = leafValues

	if numLeaves == 1 {
		// special case - constant value tree
		return t, nil
	}

	leftChilds, err := params.ToInt32Slice("left_child")
	if err != nil {
		return t, err
	}
	rightChilds, err := params.ToInt32Slice("right_child")
	if err != nil {
		return t, err
	}
	decisionTypes, err := params.ToUint32Slice("decision_type")
	if err != nil {
		return t, err
	}
	splitFeatures, err := params.ToUint32Slice("split_feature")
	if err != nil {
		return t, err
	}
	thresholds, err := params.ToFloat64Slice("threshold")
	if err != nil {
		return t, err
	}

	catThresholds := make([]uint32, 0)
	catBoundaries := make([]uint32, 0)
	if numCategorical > 0 {
		// first element set to zero for consistency
		t.catBoundaries = make([]uint32, 1)
		catThresholds, err = params.ToUint32Slice("cat_threshold")
		if err != nil {
			return t, err
		}
		catBoundaries, err = params.ToUint32Slice("cat_boundaries")
		if err != nil {
			return t, err
		}
	}

	createNumericalNode := func(idx int32) (lgNode, error) {
		node := lgNode{}
		missingType, err := convertMissingType(decisionTypes[idx])
		if err != nil {
			return node, err
		}
		defaultType := uint8(0)
		if decisionTypes[idx]&(1<<1) > 0 {
			defaultType = defaultLeft
		}
		node = numericalNode(splitFeatures[idx], missingType, thresholds[idx], defaultType)
		if leftChilds[idx] < 0 {
			node.Flags |= leftLeaf
			node.Left = uint32(^leftChilds[idx])
		}
		if rightChilds[idx] < 0 {
			node.Flags |= rightLeaf
			node.Right = uint32(^rightChilds[idx])
		}
		return node, nil
	}

	createCategoricalNode := func(idx int32) (lgNode, error) {
		node := lgNode{}
		missingType, err := convertMissingType(decisionTypes[idx])
		if err != nil {
			return node, err
		}

		catIdx := uint32(thresholds[idx])
		catType := uint8(0)
		bitsetSize := catBoundaries[catIdx+1] - catBoundaries[catIdx]
		thresholdSlice := catThresholds[catBoundaries[catIdx]:catBoundaries[catIdx+1]]
		nBits := util.NumberOfSetBits(thresholdSlice)
		if nBits == 0 {
			return node, fmt.Errorf("no bits set")
		} else if nBits == 1 {
			i, err := util.FirstNonZeroBit(thresholdSlice)
			if err != nil {
				return node, fmt.Errorf("not reached error")
			}
			catIdx = i
			catType = catOneHot
		} else if bitsetSize == 1 {
			catIdx = catThresholds[catBoundaries[catIdx]]
			catType = catSmall
		} else {
			// regular case with large bitset
			catIdx = uint32(len(t.catBoundaries) - 1)
			t.catThresholds = append(t.catThresholds, thresholdSlice...)
			t.catBoundaries = append(t.catBoundaries, uint32(len(t.catThresholds)))
		}

		node = categoricalNode(splitFeatures[idx], missingType, catIdx, catType)
		if leftChilds[idx] < 0 {
			node.Flags |= leftLeaf
			node.Left = uint32(^leftChilds[idx])
		}
		if rightChilds[idx] < 0 {
			node.Flags |= rightLeaf
			node.Right = uint32(^rightChilds[idx])
		}
		return node, nil
	}
	createNode := func(idx int32) (lgNode, error) {
		if decisionTypes[idx]&1 > 0 {
			return createCategoricalNode(idx)
		}
		return createNumericalNode(idx)
	}
	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, numNodes)
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
			origIdx := rightChilds[origNodeIdxStack[len(origNodeIdxStack)-1]]
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
			origIdx := leftChilds[origNodeIdxStack[len(origNodeIdxStack)-1]]
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

// LGEnsembleFromReader reads LightGBM model from `reader`
func LGEnsembleFromReader(reader *bufio.Reader) (*Ensemble, error) {
	e := &lgEnsemble{name: "lightgbm.gbdt"}
	params, err := util.ReadParamsUntilBlank(reader)
	if err != nil {
		return nil, err
	}

	if err := params.Compare("version", "v2"); err != nil {
		return nil, err
	}
	nClasses, err := params.ToInt("num_class")
	if err != nil {
		return nil, err
	}
	nTreePerIteration, err := params.ToInt("num_tree_per_iteration")
	if err != nil {
		return nil, err
	}
	if nClasses != nTreePerIteration {
		return nil, fmt.Errorf("meet case when num_class (%d) != num_tree_per_iteration (%d)", nClasses, nTreePerIteration)
	} else if nClasses < 1 {
		return nil, fmt.Errorf("num_class (%d) should be > 0", nClasses)
	} else if nTreePerIteration < 1 {
		return nil, fmt.Errorf("num_tree_per_iteration (%d) should be > 0", nTreePerIteration)
	}
	e.nClasses = nClasses

	maxFeatureIdx, err := params.ToInt("max_feature_idx")
	if err != nil {
		return nil, err
	}
	e.MaxFeatureIdx = maxFeatureIdx

	if params.Contains("average_output") {
		e.name = "lightgbm.rf"
		e.averageOutput = true
	}

	treeSizesStr, isFound := params["tree_sizes"]
	if !isFound {
		return nil, fmt.Errorf("no tree_sizes field")
	}
	treeSizes := strings.Split(treeSizesStr, " ")

	nTrees := len(treeSizes)
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file (based on tree_sizes value)")
	} else if nTrees%e.nClasses != 0 {
		return nil, fmt.Errorf("wrong number of trees (%d) for number of class (%d)", nTrees, e.nClasses)
	}

	e.Trees = make([]lgTree, 0, nTrees)
	for i := 0; i < nTrees; i++ {
		tree, err := lgTreeFromReader(reader)
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
	}
	return &Ensemble{e}, nil
}

// LGEnsembleFromFile reads LightGBM model from binary file
func LGEnsembleFromFile(filename string) (*Ensemble, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return LGEnsembleFromReader(bufReader)
}
