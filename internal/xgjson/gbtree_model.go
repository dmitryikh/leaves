package xgjson

import "github.com/dmitryikh/leaves/internal/xgbin"

type GBTreeJson struct {
	Learner GBTreeLearner `json:"learner"`
	Version []int         `json:"version"`
}

type GBTreeLearner struct {
	FeatureNames      []string                      `json:"feature_names"`
	FeatureTypes      []string                      `json:"feature_types"`
	GradientBooster   GBTreeBooster                 `json:"gradient_booster"`
	Objective         Objective                     `json:"objective"`
	LearnerModelParam xgbin.LearnerModelParamLegacy `json:"learner_model_param"`
}

type GBTreeBooster struct {
	Model      GBTreeModel `json:"model"`
	WeightDrop []float64   `json:"weight_drop"`
	Name       string      `json:"name"`
}

type GBTreeModel struct {
	GbTreeModelParam xgbin.GBTreeModelParam `json:"gbtree_model_param"`
	Trees            []*Tree                `json:"trees"`
	TreeInfo         []int32                `json:"tree_info"`
}

type Tree struct {
	TreeParam          xgbin.TreeParam `json:"tree_param"`
	Id                 int             `json:"id"`
	LossChanges        []float32       `json:"loss_changes"`
	SumHessian         []float32       `json:"sum_hessian"`
	BaseWeights        []float32       `json:"base_weights"`
	LeftChildren       []int32         `json:"left_children"`
	RightChildren      []int32         `json:"right_children"`
	Parents            []int32         `json:"parents"`
	SplitIndices       []uint32        `json:"split_indices"`
	SplitConditions    []float32       `json:"split_conditions"`
	SplitType          []int32         `json:"split_type"`
	DefaultLeft        []bool          `json:"default_left"`
	Categories         []int32         `json:"categories"`
	CategoriesNodes    []int32         `json:"categories_nodes"`
	CategoriesSegments []int32         `json:"categories_segments"`
	CategoricalSizes   []int32         `json:"categorical_sizes"`
}

func (g *GBTreeModel) ToBinGBTreeModel() *xgbin.GBTreeModel {
	param := g.GbTreeModelParam
	trees := make([]*xgbin.TreeModel, param.NumTrees)
	for idx, tree := range g.Trees {
		trees[idx] = tree.toBinTreeModel()
	}
	treeInfo := g.TreeInfo
	gbTreeModel := &xgbin.GBTreeModel{
		Param:    param,
		Trees:    trees,
		TreeInfo: treeInfo,
	}
	return gbTreeModel
}

func (t *Tree) toBinTreeModel() *xgbin.TreeModel {
	nodes := make([]xgbin.Node, t.TreeParam.NumNodes)
	rTreeNodeStat := make([]xgbin.RTreeNodeStat, t.TreeParam.NumNodes)
	for idx := range nodes {
		nodes[idx].CRight = t.RightChildren[idx]
		nodes[idx].CLeft = t.LeftChildren[idx]
		nodes[idx].Parent = t.Parents[idx]
		nodes[idx].Parent = int32(uint32(t.Parents[idx]) | 1 << 31)
		if t.DefaultLeft[idx] {
			t.SplitIndices[idx] |= 1 << 31
		}
		nodes[idx].SIndex = t.SplitIndices[idx]
		nodes[idx].Info = t.SplitConditions[idx]
		rTreeNodeStat[idx].BaseWeight = t.BaseWeights[idx]
		rTreeNodeStat[idx].LossChg = t.LossChanges[idx]
		rTreeNodeStat[idx].SumHess = t.SumHessian[idx]
	}
	treeParam := t.TreeParam
	treeParam.NumRoots = 1
	treeModel := &xgbin.TreeModel{
		Nodes: nodes,
		Stats: rTreeNodeStat,
		Param: treeParam,
	}
	return treeModel
}
