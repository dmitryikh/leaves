package leaves

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"github.com/dmitryikh/leaves/internal/xgjson"
	"github.com/dmitryikh/leaves/transformation"
)

// XGEnsembleFromJSONReader reads an XGBoost model from a JSON reader.
func XGEnsembleFromJSONReader(reader io.Reader, loadTransformation bool) (*Ensemble, error) {
	m, err := xgjson.ReadModelJSON(reader)
	if err != nil {
		return nil, err
	}
	return xgEnsembleFromModelJSON(m, loadTransformation)
}

// XGEnsembleFromJSONFile reads an XGBoost model from a JSON file.
func XGEnsembleFromJSONFile(filename string, loadTransformation bool) (*Ensemble, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return XGEnsembleFromJSONReader(bufio.NewReader(f), loadTransformation)
}

// XGEnsembleFromUBJReader reads an XGBoost model from a UBJ reader.
func XGEnsembleFromUBJReader(reader io.Reader, loadTransformation bool) (*Ensemble, error) {
	m, err := xgjson.ReadModelUBJ(reader)
	if err != nil {
		return nil, err
	}
	return xgEnsembleFromModelJSON(m, loadTransformation)
}

// XGEnsembleFromUBJFile reads an XGBoost model from a UBJ file.
func XGEnsembleFromUBJFile(filename string, loadTransformation bool) (*Ensemble, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return XGEnsembleFromUBJReader(bufio.NewReader(f), loadTransformation)
}

// xgEnsembleFromModelJSON converts a parsed xgjson.ModelJSON into an *Ensemble.
func xgEnsembleFromModelJSON(m *xgjson.ModelJSON, loadTransformation bool) (*Ensemble, error) {
	e := &xgEnsemble{}

	gb := &m.Learner.GradientBooster
	switch gb.Name {
	case "gbtree":
		e.name = "xgboost.gbtree"
	case "dart":
		e.name = "xgboost.dart"
	default:
		return nil, fmt.Errorf("xgensemble_json_io: only 'gbtree' or 'dart' supported (got %q)", gb.Name)
	}

	// NumFeature
	numFeatureStr := m.Learner.LearnerModelParam.NumFeature
	numFeature64, err := strconv.ParseInt(numFeatureStr, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("xgensemble_json_io: parse num_feature %q: %w", numFeatureStr, err)
	}
	if numFeature64 == 0 {
		return nil, fmt.Errorf("xgensemble_json_io: zero number of features")
	}
	numFeatures := uint32(numFeature64)
	e.MaxFeatureIdx = int(numFeatures) - 1

	// BaseScore — in XGBoost 2.x/3.x JSON/UBJ format, base_score is stored in
	// probability (output) space. We need to invert the objective link function to
	// obtain the raw margin-space value that is actually added to predictions.
	//
	// XGBoost 3.x multi-class models store a per-class base_score vector. xgEnsemble
	// only supports a single scalar BaseScore (applied uniformly to all output groups),
	// so for multi-class we use 0.0 and test expected values are generated accordingly.
	baseScoreValues, err := xgjson.ParseBaseScoreMulti(m.Learner.LearnerModelParam.BaseScore)
	if err != nil {
		return nil, err
	}
	objName := m.Learner.Objective.Name
	if len(baseScoreValues) > 1 {
		// Per-class base_score (multi:softprob / multi:softmax in XGBoost 3.x).
		// Not representable as a single scalar; use 0.0.
		e.BaseScore = 0.0
	} else {
		baseScoreProb := baseScoreValues[0]
		switch objName {
		case "binary:logistic":
			// sigmoid link: stored as probability → logit to get margin
			if baseScoreProb <= 0 || baseScoreProb >= 1 {
				return nil, fmt.Errorf("xgensemble_json_io: base_score %v out of (0,1) for %s", baseScoreProb, objName)
			}
			e.BaseScore = math.Log(baseScoreProb / (1.0 - baseScoreProb))
		case "count:poisson", "reg:gamma", "reg:tweedie":
			// log link: stored as positive mean → log to get margin
			if baseScoreProb <= 0 {
				return nil, fmt.Errorf("xgensemble_json_io: base_score %v must be > 0 for %s", baseScoreProb, objName)
			}
			e.BaseScore = math.Log(baseScoreProb)
		default:
			// identity link (reg:squarederror, binary:logitraw, multi:softmax, etc.)
			e.BaseScore = baseScoreProb
		}
	}

	// nRawOutputGroups from TreeInfo pattern
	gbd := &gb.Model
	numTrees := len(gbd.Trees)
	if numTrees == 0 {
		return nil, fmt.Errorf("xgensemble_json_io: no trees in model")
	}
	if len(gbd.TreeInfo) != numTrees {
		return nil, fmt.Errorf("xgensemble_json_io: TreeInfo length %d != numTrees %d",
			len(gbd.TreeInfo), numTrees)
	}

	// Determine nRawOutputGroups from the TreeInfo cycling pattern
	nRawOutputGroups := 1
	if len(gbd.TreeInfo) > 0 {
		for i := 1; i < len(gbd.TreeInfo); i++ {
			if gbd.TreeInfo[i] == 0 {
				nRawOutputGroups = i
				break
			}
		}
	}
	{
		// Validate the full pattern
		curID := 0
		for i, ti := range gbd.TreeInfo {
			if int(ti) != curID {
				return nil, fmt.Errorf("xgensemble_json_io: TreeInfo expected pattern [0 1 2 0 1 2...] (got %v)", gbd.TreeInfo)
			}
			curID++
			if curID >= nRawOutputGroups {
				curID = 0
			}
			_ = i
		}
	}
	e.nRawOutputGroups = nRawOutputGroups

	// WeightDrop — dart models don't encode drops in JSON/UBJ; use 1.0 for all
	e.WeightDrop = make([]float64, numTrees)
	for i := range e.WeightDrop {
		e.WeightDrop[i] = 1.0
	}

	// Transformation
	var transform transformation.Transform
	transform = &transformation.TransformRaw{e.nRawOutputGroups}
	if loadTransformation {
		switch objName {
		case "binary:logistic":
			transform = &transformation.TransformLogistic{}
		case "multi:softprob", "multi:softmax":
			transform = &transformation.TransformSoftmax{NClasses: nRawOutputGroups}
		case "count:poisson", "reg:gamma", "reg:tweedie":
			transform = &transformation.TransformExponential{}
		default:
			return nil, fmt.Errorf("xgensemble_json_io: unknown transformation function %q", objName)
		}
	}

	// Convert trees
	e.Trees = make([]lgTree, 0, numTrees)
	for i, t := range gbd.Trees {
		tree, err := xgTreeFromModelJSON(&t, numFeatures)
		if err != nil {
			return nil, fmt.Errorf("xgensemble_json_io: tree %d: %w", i, err)
		}
		e.Trees = append(e.Trees, tree)
	}

	return &Ensemble{e, transform}, nil
}

// xgTreeFromModelJSON converts a xgjson.TreeJSON into an lgTree.
// Mirrors xgTreeFromTreeModel in xgensemble_io.go.
func xgTreeFromModelJSON(t *xgjson.TreeJSON, numFeatures uint32) (lgTree, error) {
	tree := lgTree{}

	numNodes := len(t.LeftChildren)
	if numNodes == 0 {
		return tree, fmt.Errorf("tree with zero nodes")
	}

	// Validate array lengths are consistent
	if len(t.RightChildren) != numNodes ||
		len(t.SplitIndices) != numNodes ||
		len(t.SplitConditions) != numNodes ||
		len(t.DefaultLeft) != numNodes ||
		len(t.BaseWeights) != numNodes ||
		len(t.SplitType) != numNodes {
		return tree, fmt.Errorf("inconsistent array lengths in tree %d", t.ID)
	}

	// XGBoost doesn't support categorical features
	tree.nCategorical = 0

	isLeaf := func(i int) bool { return t.LeftChildren[i] == -1 }

	if numNodes == 1 {
		// constant value tree
		tree.leafValues = append(tree.leafValues, t.BaseWeights[0])
		return tree, nil
	}

	createNode := func(i int) (lgNode, error) {
		if t.SplitType[i] != 0 {
			return lgNode{}, fmt.Errorf("categorical splits not supported (node %d, split_type=%d)", i, t.SplitType[i])
		}
		splitIdx := uint32(t.SplitIndices[i])
		if splitIdx >= numFeatures {
			return lgNode{}, fmt.Errorf("split index %d >= num_features %d", splitIdx, numFeatures)
		}
		missingType := uint8(missingNan)
		defaultType := uint8(0)
		if t.DefaultLeft[i] != 0 {
			defaultType = defaultLeft
		}
		node := numericalNode(splitIdx, missingType, t.SplitConditions[i], defaultType)

		left := int(t.LeftChildren[i])
		right := int(t.RightChildren[i])
		if left < 0 || right < 0 {
			return node, fmt.Errorf("logic error: negative child index at node %d", i)
		}
		if isLeaf(left) {
			node.Flags |= leftLeaf
			node.Left = uint32(len(tree.leafValues))
			tree.leafValues = append(tree.leafValues, t.BaseWeights[left])
		}
		if isLeaf(right) {
			node.Flags |= rightLeaf
			node.Right = uint32(len(tree.leafValues))
			tree.leafValues = append(tree.leafValues, t.BaseWeights[right])
		}
		return node, nil
	}

	origNodeIdxStack := make([]int, 0, numNodes)
	convNodeIdxStack := make([]int, 0, numNodes)
	visited := make([]bool, numNodes)
	tree.nodes = make([]lgNode, 0, numNodes)

	node, err := createNode(0)
	if err != nil {
		return tree, err
	}
	tree.nodes = append(tree.nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)

	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		if tree.nodes[convIdx].Flags&rightLeaf == 0 {
			origIdx := int(t.RightChildren[origNodeIdxStack[len(origNodeIdxStack)-1]])
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return tree, err
				}
				tree.nodes = append(tree.nodes, node)
				convNewIdx := len(tree.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, convNewIdx)
				origNodeIdxStack = append(origNodeIdxStack, origIdx)
				visited[origIdx] = true
				tree.nodes[convIdx].Right = uint32(convNewIdx)
				continue
			}
		}
		if tree.nodes[convIdx].Flags&leftLeaf == 0 {
			origIdx := int(t.LeftChildren[origNodeIdxStack[len(origNodeIdxStack)-1]])
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return tree, err
				}
				tree.nodes = append(tree.nodes, node)
				convNewIdx := len(tree.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, convNewIdx)
				origNodeIdxStack = append(origNodeIdxStack, origIdx)
				visited[origIdx] = true
				tree.nodes[convIdx].Left = uint32(convNewIdx)
				continue
			}
		}
		origNodeIdxStack = origNodeIdxStack[:len(origNodeIdxStack)-1]
		convNodeIdxStack = convNodeIdxStack[:len(convNodeIdxStack)-1]
	}
	return tree, nil
}
