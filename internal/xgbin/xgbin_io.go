package xgbin

import (
	"bufio"
	"encoding/binary"
)

// Most data structures from this packages are mirrors from original XGBoost
// data structures.
// Note: XGBosst widely use `bst_float` typedef type which equals to `float` (float32 in Go) for now
// Note: XGBosst widely use int type which is machine depended. Go's int32 should cover most common case
// Note: Data structures' fields comments are take from original XGBoost source code

// LearnerModelParam - training parameter for regression.
// from src/learner.cc
type LearnerModelParam struct {
	// global bias
	BaseScore float32
	// number of features
	NumFeatures uint32
	// number of classes, if it is multi-class classification
	NumClass int32
	// Model contain additional properties
	ContainExtraAttrs int32
	// Model contain eval metrics
	ContainEvalMetrics int32
	// reserved field
	Reserved [29]int32
}

// GBTreeModelParam - model parameters
// from src/gbm/gbtree_model.h
type GBTreeModelParam struct {
	// number of trees
	NumTrees int32
	// number of roots
	NumRoots int32
	// number of features to be used by trees
	NumFeature int32
	// pad this space, for backward compatibility reason
	Pad32bit int32
	// deprecated padding space.
	NumPbufferDeprecated int64
	// how many output group a single instance can produce
	// this affects the behavior of number of output we have:
	// suppose we have n instance and k group, output will be k * n
	NumOutputGroup int32
	// size of leaf vector needed in tree
	SizeLeafVector int32
	// reserved parameters
	Reserved [32]int32
}

// TreeParam - meta parameters of the tree
// from include/xgboost/tree_model.h
type TreeParam struct {
	// number of start root
	NumRoots int32
	// total number of nodes
	NumNodes int32
	// number of deleted nodes
	NumDeleted int32
	// maximum depth, this is a statistics of the tree
	MaxDepth int32
	// number of features used for tree construction
	NumFeature int32
	// leaf vector size, used for vector tree
	// used to store more than one dimensional information in tree
	SizeLeafVector int32
	// reserved part, make sure alignment works for 64bit
	Reserved [31]int32
}

// Node - tree Node for XGBoost's RegTree class
// from include/xgboost/tree_model.h
type Node struct {
	// pointer to parent, highest bit is used to
	// indicate whether it's a left child or not
	Parent int32
	// pointer to left, right
	// NOTE: CLeft == -1 means leaf node
	CLeft  int32
	CRight int32
	// split feature index, left split or right split depends on the highest bit
	SIndex uint32
	// extra info
	// union Info{
	//   bst_float leaf_value;
	//   TSplitCond split_cond;
	// };
	Info float32
}

// RTreeNodeStat - node statistics used in regression tree
// from include/xgboost/tree_model.h
type RTreeNodeStat struct {
	// loss change caused by current split
	LossChg float32
	// sum of hessian values, used to measure coverage of data
	SumHess float32
	// weight of current node
	BaseWeight float32
	// number of child that is leaf node known up to now
	LeafChildCnt int32
}

// GBLinearModelParam - model parameters
// from src/gbm/gblinear_model.h
type GBLinearModelParam struct {
	// number of feature dimension
	NumFeature uint32
	// number of output group
	NumOutputGroup int32
	// reserved field
	Reserved [32]int32
}

// TreeModel contains all input data related to particular tree. Used just as
// a container of input data for go implementation. Objects layout could be
// arbitrary
type TreeModel struct {
	Nodes []Node
	Stats []RTreeNodeStat
	// // leaf vector, that is used to store additional information
	// LeafVector []float32
	Param TreeParam
}

// GBTreeModel contains all input data related to gbtree model. Used just as a
// container of input data for go implementation. Objects layout could be
// arbitrary
type GBTreeModel struct {
	Param GBTreeModelParam
	Trees []*TreeModel
	// some information indicator of the tree, reserved
	TreeInfo []int32
}

// ModelHeader contains all input data related to top records of model binary
// file. Used just as a container of input data for go implementation. Objects
// layout could be arbitrary
type ModelHeader struct {
	Param   LearnerModelParam
	NameObj string
	NameGbm string
}

// GBLinearModel contains all data about gblinear model read from binary file.
// Used just as a container of input data for go implementation. Objects
// layout could be arbitrary
type GBLinearModel struct {
	Param   GBLinearModelParam
	Weights []float32
}

// ReadStruct - read arbitrary data structure from binary stream
func ReadStruct(reader *bufio.Reader, dst interface{}) error {
	err := binary.Read(reader, binary.LittleEndian, dst)
	if err != nil {
		return err
	}
	return nil
}

// ReadString - read ascii string from binary stream
// from dmlc-core/include/dmlc/serializer.h
func ReadString(reader *bufio.Reader) (string, error) {
	var size uint64
	err := binary.Read(reader, binary.LittleEndian, &size)
	if err != nil {
		return "", err
	}
	if size == 0 {
		return "", nil
	}
	bytes := make([]byte, size)
	err = binary.Read(reader, binary.LittleEndian, &bytes)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

// ReadFloat32Slice - read vector of floats from binary stream
// from dmlc-core/include/dmlc/serializer.h
func ReadFloat32Slice(reader *bufio.Reader) ([]float32, error) {
	var size uint64
	err := binary.Read(reader, binary.LittleEndian, &size)
	if err != nil {
		return nil, err
	}
	if size == 0 {
		return nil, nil
	}
	vec := make([]float32, size)
	err = binary.Read(reader, binary.LittleEndian, &vec)
	if err != nil {
		return nil, err
	}
	return vec, nil
}

// ReadInt32Slice - read vector of int from binary stream
// from dmlc-core/include/dmlc/serializer.h
func ReadInt32Slice(reader *bufio.Reader) ([]int32, error) {
	var size uint64
	err := binary.Read(reader, binary.LittleEndian, &size)
	if err != nil {
		return nil, err
	}
	if size == 0 {
		return nil, nil
	}
	vec := make([]int32, size)
	err = binary.Read(reader, binary.LittleEndian, &vec)
	if err != nil {
		return nil, err
	}
	return vec, nil
}

// ReadModelHeader reads header info from binary model file
func ReadModelHeader(reader *bufio.Reader) (*ModelHeader, error) {
	modelHeader := &ModelHeader{}
	err := ReadStruct(reader, &modelHeader.Param)
	if err != nil {
		return nil, err
	}

	nameObj, err := ReadString(reader)
	if err != nil {
		return nil, err
	}
	modelHeader.NameObj = nameObj

	nameGbm, err := ReadString(reader)
	if err != nil {
		return nil, err
	}
	modelHeader.NameGbm = nameGbm
	return modelHeader, nil
}

// ReadGBTreeModel reads gbtree model from binary model file
func ReadGBTreeModel(reader *bufio.Reader) (*GBTreeModel, error) {
	gBTreeModel := &GBTreeModel{}
	err := ReadStruct(reader, &gBTreeModel.Param)
	if err != nil {
		return nil, err
	}

	for i := int32(0); i < gBTreeModel.Param.NumTrees; i++ {
		tree, err := ReadTreeModel(reader)
		if err != nil {
			return nil, err
		}
		gBTreeModel.Trees = append(gBTreeModel.Trees, tree)
	}
	if gBTreeModel.Param.NumTrees > 0 {
		// some information indicator of the tree, reserved
		// std::vector<int> tree_info;
		gBTreeModel.TreeInfo = make([]int32, gBTreeModel.Param.NumTrees)
		err = binary.Read(reader, binary.LittleEndian, &gBTreeModel.TreeInfo)
		if err != nil {
			return nil, err
		}
	}
	// NOTE: skip other attributes in binary format, because we don't need them
	return gBTreeModel, nil
}

// ReadTreeModel reads particular tree data from binary model file
func ReadTreeModel(reader *bufio.Reader) (*TreeModel, error) {
	treeModel := &TreeModel{}
	err := ReadStruct(reader, &treeModel.Param)
	if err != nil {
		return nil, err
	}
	treeModel.Nodes = make([]Node, 0, treeModel.Param.NumNodes)
	treeModel.Stats = make([]RTreeNodeStat, 0, treeModel.Param.NumNodes)
	for i := int32(0); i < treeModel.Param.NumNodes; i++ {
		node := Node{}
		err := ReadStruct(reader, &node)
		if err != nil {
			return nil, err
		}
		treeModel.Nodes = append(treeModel.Nodes, node)
	}
	for i := int32(0); i < treeModel.Param.NumNodes; i++ {
		stat := RTreeNodeStat{}
		err := ReadStruct(reader, &stat)
		if err != nil {
			return nil, err
		}
		treeModel.Stats = append(treeModel.Stats, stat)
	}
	if treeModel.Param.SizeLeafVector > 0 {
		// leaf vector, that is used to store additional information
		// std::vector<bst_float> leaf_vector_;
		_, err := ReadFloat32Slice(reader)
		if err != nil {
			return nil, err
		}
	}
	return treeModel, nil
}

// ReadGBLinearModel reads gblinear model from binary model file
func ReadGBLinearModel(reader *bufio.Reader) (*GBLinearModel, error) {
	gbLinearModel := &GBLinearModel{}
	err := ReadStruct(reader, &gbLinearModel.Param)
	if err != nil {
		return nil, err
	}
	gbLinearModel.Weights, err = ReadFloat32Slice(reader)
	if err != nil {
		return nil, err
	}
	return gbLinearModel, nil
}
