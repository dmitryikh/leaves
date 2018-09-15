package leaves

import (
	"bufio"
	"fmt"
	"strings"
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

func lgTreeFromReader(reader *bufio.Reader) (LGTree, error) {
	t := LGTree{}
	params, err := readParamsUntilBlank(reader)
	if err != nil {
		return t, err
	}
	numCategorical, err := mapValueToInt(params, "num_cat")
	if err != nil {
		return t, err
	}
	t.nCategorical = uint32(numCategorical)

	numLeaves, err := mapValueToInt(params, "num_leaves")
	if err != nil {
		return t, err
	}
	if numLeaves < 2 {
		return t, fmt.Errorf("num_leaves < 2")
	}
	numNodes := numLeaves - 1

	leafValues, err := mapValueToFloat64Slice(params, "leaf_value")
	if err != nil {
		return t, err
	}
	t.leafValues = leafValues
	leftChilds, err := mapValueToInt32Slice(params, "left_child")
	if err != nil {
		return t, err
	}
	rightChilds, err := mapValueToInt32Slice(params, "right_child")
	if err != nil {
		return t, err
	}
	decisionTypes, err := mapValueToUint32Slice(params, "decision_type")
	if err != nil {
		return t, err
	}
	splitFeatures, err := mapValueToUint32Slice(params, "split_feature")
	if err != nil {
		return t, err
	}
	thresholds, err := mapValueToFloat64Slice(params, "threshold")
	if err != nil {
		return t, err
	}

	catThresholds := make([]uint32, 0)
	catBoundaries := make([]uint32, 0)
	if numCategorical > 0 {
		// first element set to zero for consistency
		t.catBoundaries = make([]uint32, 1)
		catThresholds, err = mapValueToUint32Slice(params, "cat_threshold")
		if err != nil {
			return t, err
		}
		catBoundaries, err = mapValueToUint32Slice(params, "cat_boundaries")
		if err != nil {
			return t, err
		}
	}

	createNumericalNode := func(idx int32) (LGNode, error) {
		node := LGNode{}
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

	createCategoricalNode := func(idx int32) (LGNode, error) {
		node := LGNode{}
		missingType, err := convertMissingType(decisionTypes[idx])
		if err != nil {
			return node, err
		}

		catIdx := uint32(thresholds[idx])
		catType := uint8(0)
		bitsetSize := catBoundaries[catIdx+1] - catBoundaries[catIdx]
		thresholdSlice := catThresholds[catBoundaries[catIdx]:catBoundaries[catIdx+1]]
		nBits := numberOfSetBits(thresholdSlice)
		if nBits == 0 {
			return node, fmt.Errorf("no bits set")
		} else if nBits == 1 {
			i, err := firstNonZeroBit(thresholdSlice)
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
	createNode := func(idx int32) (LGNode, error) {
		if decisionTypes[idx]&1 > 0 {
			return createCategoricalNode(idx)
		}
		return createNumericalNode(idx)
	}
	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, numNodes)
	t.nodes = make([]LGNode, 0, numNodes)
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
func LGEnsembleFromReader(reader *bufio.Reader) (*LGEnsemble, error) {
	e := &LGEnsemble{}
	params, err := readParamsUntilBlank(reader)
	if err != nil {
		return nil, err
	}

	if err := mapValueCompare(params, "version", "v2"); err != nil {
		return nil, err
	}
	if err := mapValueCompare(params, "num_class", "1"); err != nil {
		return nil, err
	}
	if err := mapValueCompare(params, "num_tree_per_iteration", "1"); err != nil {
		return nil, err
	}
	if maxFeatureIdx, err := mapValueToInt(params, "max_feature_idx"); err != nil {
		return nil, err
	} else {
		e.MaxFeatureIdx = uint32(maxFeatureIdx)
	}

	treeSizesStr, isFound := params["tree_sizes"]
	if !isFound {
		return nil, fmt.Errorf("no tree_sizes field")
	}
	treeSizes := strings.Split(treeSizesStr, " ")

	nTrees := len(treeSizes)
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file (based on tree_sizes value)")
	}

	e.Trees = make([]LGTree, 0, nTrees)
	for i := 0; i < nTrees; i++ {
		tree, err := lgTreeFromReader(reader)
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		e.Trees = append(e.Trees, tree)
	}
	return e, nil
}
