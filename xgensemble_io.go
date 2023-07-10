package leaves

import (
	"bufio"
	"fmt"
	"math"
	"os"

	"github.com/dmitryikh/leaves/internal/xgbin"
	"github.com/dmitryikh/leaves/internal/xgjson"
	"github.com/dmitryikh/leaves/transformation"
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
		// special case - constant value tree
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

func readWeightDropFromReader(reader *bufio.Reader, numTrees int, modelName string) ([]float64, error) {
	origModelWeightDrop := make([]float64, numTrees)
	if err := checkModelName(modelName); err != nil {
		return nil, err
	}
	if modelName == "dart" {
		// read additional float32 slice of weighs of dropped trees. Only for 'dart' models
		weightDrop, err := xgbin.ReadFloat32Slice(reader)
		if err != nil {
			return nil, err
		}
		if len(weightDrop) != numTrees {
			return nil, fmt.Errorf(
				"unexpected len(weightDrop) for 'dart' (got: %d, expected: %d)",
				len(weightDrop),
				numTrees,
			)
		}
		for i, v := range weightDrop {
			origModelWeightDrop[i] = float64(v)
		}
	} else if modelName == "gbtree" {
		// use 1.0 as default. 1.0 scale will not break down anything
		for i := 0; i < numTrees; i++ {
			origModelWeightDrop[i] = 1.0
		}
	}
	return origModelWeightDrop, nil
}

// XGEnsembleFromReader reads XGBoost model from `reader`. Works with 'gbtree' and 'dart' models
func XGEnsembleFromReader(reader *bufio.Reader, loadTransformation bool) (*Ensemble, error) {
	e := &xgEnsemble{}

	//to support version after 1.0.0
	xgbin.ReadBinf(reader)
	// reading header info
	header, err := xgbin.ReadModelHeader(reader)
	if err != nil {
		return nil, err
	}
	// reading gbtree
	origModel, err := xgbin.ReadGBTreeModel(reader)
	if err != nil {
		return nil, err
	}
	if e.name, err = createModelName(header.NameGbm); err != nil {
		return nil, err
	}
	if e.MaxFeatureIdx, err = calculateMaxFeatureIdx(int(header.Param.NumFeatures)); err != nil {
		return nil, err
	}
	//To support version before 1.0.0
	if header.Param.MajorVersion > uint32(0) {
		e.nRawOutputGroups = getNRawOutputGroups(header.Param.NumClass)
		e.BaseScore = calculateBaseScoreFromLearnerParam(float64(header.Param.BaseScore))
	} else {
		e.nRawOutputGroups = getNRawOutputGroups(origModel.Param.DeprecatedNumOutputGroup)
		e.BaseScore = float64(header.Param.BaseScore)
	}
	if origModel.Param.DeprecatedNumFeature > int32(header.Param.NumFeatures) {
		return nil, fmt.Errorf(
			"gbtee number of features %d, but header number of features %d",
			origModel.Param.DeprecatedNumFeature,
			header.Param.NumFeatures,
		)
	}

	if e.WeightDrop, err = readWeightDropFromReader(reader, int(origModel.Param.NumTrees), header.NameGbm); err != nil {
		return nil, err
	}
	// TODO: below is not true (see Agaricus test). Why?
	// if header.GbTreeModelParam.NumClass != origModel.GbTreeModelParam.DeprecatedNumOutputGroup {
	// 	return nil, fmt.Errorf("header number of class and model number of class should be the same (%d != %d)",
	// 		header.GbTreeModelParam.NumClass, origModel.GbTreeModelParam.DeprecatedNumOutputGroup)
	// }
	if origModel.Param.DeprecatedNumRoots != 1 {
		return nil, fmt.Errorf("support only trees with 1 root (got %d)", origModel.Param.DeprecatedNumRoots)
	}
	if len(origModel.TreeInfo) != int(origModel.Param.NumTrees) {
		return nil, fmt.Errorf("TreeInfo size should be %d (got %d)",
			int(origModel.Param.NumTrees),
			len(origModel.TreeInfo))
	}
	if err = checkTreeInfo(origModel.TreeInfo, e.nRawOutputGroups); err != nil {
		return nil, err
	}

	transform, err := createTransform(loadTransformation, e.nRawOutputGroups, header.NameObj)
	if err != nil {
		return nil, err
	}
	if e.Trees, err = createTrees(origModel.Param.NumTrees, header.Param.NumFeatures, origModel); err != nil {
		return nil, err
	}
	return &Ensemble{e, transform}, nil
}

func checkTreeInfo(treeInfo []int32, nRawOutputGroups int) error {
	// Check that TreeInfo has expected pattern (0 1 2 0 1 2...)
	curID := 0
	for i := 0; i < len(treeInfo); i++ {
		if int(treeInfo[i]) != curID {
			return fmt.Errorf("TreeInfo expected to have pattern [0 1 2 0 1 2...] (got %v)", treeInfo)
		}
		curID++
		if curID >= nRawOutputGroups {
			curID = 0
		}
	}
	return nil
}

// XGEnsembleFromFile reads XGBoost model from binary file or json file. Works with 'gbtree' and 'dart' models
func XGEnsembleFromFile(filename string, loadTransformation bool) (*Ensemble, error) {
	if ensemble, err := xgEnsembleFromJsonFile(filename, loadTransformation); err == nil {
		return ensemble, nil
	}
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return XGEnsembleFromReader(bufReader, loadTransformation)
}

func xgEnsembleFromJsonFile(filename string, loadTransformation bool) (*Ensemble, error) {
	gbTreeJson, err := xgjson.ReadGBTree(filename)
	if err != nil {
		return nil, err
	}
	e, err := createXGEnsembleFromGBTreeJson(gbTreeJson)
	if err != nil {
		return nil, err
	}
	transform, err := createTransform(loadTransformation, e.nRawOutputGroups, gbTreeJson.Learner.Objective.Name)
	if err != nil {
		return nil, err
	}
	return &Ensemble{e, transform}, nil
}

func createXGEnsembleFromGBTreeJson(gbTreeJson *xgjson.GBTreeJson) (*xgEnsemble, error) {
	e := &xgEnsemble{}
	var err error
	e.nRawOutputGroups = getNRawOutputGroups(gbTreeJson.Learner.LearnerModelParam.NumClass)
	e.BaseScore = calculateBaseScoreFromLearnerParam(float64(gbTreeJson.Learner.LearnerModelParam.BaseScore))
	if e.MaxFeatureIdx, err = calculateMaxFeatureIdx(int(gbTreeJson.Learner.LearnerModelParam.NumFeatures)); err != nil {
		return nil, err
	}
	if e.name, err = createModelName(gbTreeJson.Learner.GradientBooster.Name); err != nil {
		return nil, err
	}
	e.WeightDrop, err = getWeightDrop(gbTreeJson)
	if err != nil {
		return nil, err
	}
	if err = checkTreeInfo(gbTreeJson.Learner.GradientBooster.Model.TreeInfo, e.nRawOutputGroups); err != nil {
		return nil, err
	}
	e.Trees, err = createTrees(
		gbTreeJson.Learner.GradientBooster.Model.GbTreeModelParam.NumTrees,
		gbTreeJson.Learner.LearnerModelParam.NumFeatures,
		gbTreeJson.Learner.GradientBooster.Model.ToBinGBTreeModel(),
	)
	if err != nil {
		return nil, err
	}
	return e, nil
}

func checkModelName(name string) error {
	switch name {
	case "dart", "gbtree":
		return nil
	default:
		return fmt.Errorf("only 'gbtree' or 'dart' is supported (got %s)", name)
	}
}

func createModelName(name string) (string, error) {
	if err := checkModelName(name); err != nil {
		return "", err
	}
	return fmt.Sprintf("xgboost.%s", name), nil
}

func calculateMaxFeatureIdx(numFeatures int) (int, error) {
	if maxFeatureIdx := numFeatures - 1; maxFeatureIdx < 0 {
		return -1, fmt.Errorf("zero number of features")
	} else {
		return maxFeatureIdx, nil
	}
}

func calculateBaseScoreFromLearnerParam(rawBaseScore float64) float64 {
	return math.Log(rawBaseScore) - math.Log(1-rawBaseScore)
}

func getNRawOutputGroups(numClass int32) int {
	nRawOutputGroups := 1
	if numClass != 0 {
		nRawOutputGroups = int(numClass)
	}
	return nRawOutputGroups
}

func getWeightDrop(gbTreeJson *xgjson.GBTreeJson) ([]float64, error) {
	weightDrop := make([]float64, gbTreeJson.Learner.GradientBooster.Model.GbTreeModelParam.NumTrees)
	if gbTreeJson.Learner.GradientBooster.Name == "dart" {
		weightDrop = gbTreeJson.Learner.GradientBooster.WeightDrop
	} else if gbTreeJson.Learner.GradientBooster.Name == "gbtree" {
		for idx := range weightDrop {
			weightDrop[idx] = 1.0
		}
	}
	return weightDrop, nil
}

func createTrees(numTrees int32, numFeatures uint32, model *xgbin.GBTreeModel) ([]lgTree, error) {
	if numTrees == 0 {
		return nil, fmt.Errorf("no trees in model")
	}
	trees := make([]lgTree, 0, numTrees)
	for i := int32(0); i < numTrees; i++ {
		tree, err := xgTreeFromTreeModel(model.Trees[i], numFeatures)
		if err != nil {
			return nil, fmt.Errorf("error while reading %d tree: %s", i, err.Error())
		}
		trees = append(trees, tree)
	}
	return trees, nil
}

func createTransform(loadTransformation bool, nRawOutputGroups int, objectiveName string) (transformation.Transform, error) {
	var transform transformation.Transform
	transform = &transformation.TransformRaw{NumOutputGroups: nRawOutputGroups}
	if loadTransformation {
		if objectiveName == "binary:logistic" {
			transform = &transformation.TransformLogistic{}
		} else {
			return nil, fmt.Errorf("unknown transformation function '%s'", objectiveName)
		}
	}
	return transform, nil
}
