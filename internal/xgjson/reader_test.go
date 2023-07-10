package xgjson

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReadGBTree(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "xgagaricus.json")
	gbTreeJson, err := ReadGBTree(path)
	assert.Nil(t, err)
	assert.Equal(t, gbTreeJson.Learner.LearnerModelParam.NumClass, int32(0))
	assert.Equal(t, gbTreeJson.Learner.LearnerModelParam.NumFeatures, uint32(127))
	assert.Equal(t, gbTreeJson.Learner.LearnerModelParam.BaseScore, float32(0.5))
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.GbTreeModelParam.NumTrees, int32(3))
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.GbTreeModelParam.SizeLeafVector, int32(0))
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Name, "gbtree")
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.TreeInfo, []int32{0, 0, 0})
	assert.Equal(t, gbTreeJson.Learner.Objective.Name, "binary:logistic")
	assert.Equal(t, gbTreeJson.Learner.Objective.RegLossParam.ScalePosWeight, "1")
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].TreeParam.NumNodes, int32(7))
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].TreeParam.NumRoots, int32(0))
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].TreeParam.NumDeleted, int32(0))
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].TreeParam.MaxDepth, int32(0))
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].TreeParam.NumFeature, int32(127))
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].TreeParam.SizeLeafVector, int32(0))
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].SplitIndices, []uint32{0x1d, 0x38, 0x6d, 0x0, 0x0, 0x0, 0x0})
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].Parents, []int32{2147483647, 0, 0, 1, 1, 2, 2})
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].DefaultLeft, []bool{true, true, true, false, false, false, false})
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].LeftChildren, []int32{1, 3, 5, -1, -1, -1, -1})
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].RightChildren, []int32{2, 4, 6, -1, -1, -1, -1})
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].LossChanges, []float32{4000.531, 1158.212, 198.17383, 0, 0, 0, 0})
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].SumHessian, []float32{1628.25, 924.5, 703.75, 812, 112.5, 690.5, 13.25})
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].SplitConditions, []float32{-9.536743e-07, -9.536743e-07, -9.536743e-07, 1.7121772, -1.7004405, -1.9407086, 1.8596492})
	assert.Equal(t, gbTreeJson.Learner.GradientBooster.Model.Trees[0].BaseWeights, []float32{-0.07150529, 1.2955159, -1.8666193, 1.7121772, -1.7004405, -1.9407086, 1.8596492})
}

func TestReadGBLinear(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "xgblin_agaricus.json")
	gbLinearJson, err := ReadGBLinear(path)
	assert.Nil(t, err)
	assert.Equal(t, gbLinearJson.Learner.LearnerModelParam.NumClass, int32(0))
	assert.Equal(t, gbLinearJson.Learner.LearnerModelParam.NumFeatures, uint32(127))
	assert.Equal(t, gbLinearJson.Learner.LearnerModelParam.BaseScore, float32(0.5))
	assert.Equal(t, gbLinearJson.Learner.GradientBooster.Model.Param.NumFeature, uint32(0))
	assert.Equal(t, gbLinearJson.Learner.GradientBooster.Model.Param.NumOutputGroup, int32(0))
	assert.Equal(t, gbLinearJson.Learner.GradientBooster.Name, "gblinear")
	assert.Equal(t, gbLinearJson.Learner.Objective.Name, "binary:logistic")
	assert.Equal(t, gbLinearJson.Learner.Objective.RegLossParam.ScalePosWeight, "1")
	assert.Equal(t, gbLinearJson.Learner.GradientBooster.Model.Weights, []float32{
		0, -3.3614023, 4.1982303, -1.6642607, -1.2977017, -0.29495186, -6.1344137, -1.6853834, 4.5176134, -0.8055269,
		1.7197797, -0.24293214, -1.2140335, -6.4709496, -1.4642721, -3.7574105, 0.12938689, -4.3361783, 0.6183258,
		-1.4268272, -1.3104084, 1.7410156, 0.117671266, -2.0570586, -2.0499654, 8.077395, 1.4196275, 0.44888568,
		4.097487, -1.1838284, 1.2735202, 1.0246072, -3.391799, 0, -1.7932326, 0, -0.33254558, 0.109187245, 0,
		-1.1411328, -0.52498317, 0.27882916, 0.03483078, 0.9740474, 0.4768914, -0.31060702, -1.3755631, 2.4963624,
		-0.29732314, -2.4207394, -3.5864277, -1.3845217, -1.9461952, 1.1676147, -0.7587152, 0.036639452, 0.74975085, 0,
		2.1593957, 0, 4.150118, -0.37633452, -0.75017554, 4.179628, 4.3672876, -0.45146954, 2.122739, -1.4882739,
		0.50152546, 0.84647465, -1.2929397, -1.120182, -1.5673592, 0.024146706, -2.536148, 0.44043186, 2.5266068,
		-0.60404134, 2.5295382e-08, -0.101869956, 4.538866, 7.651545, -4.310523, -1.0584886, -0.12134789, 1.1975892,
		-0.038880825, 9.091684, -0.56001216, 0, 0.67243505, 1.3224936, -0.11018723, -1.7897928, 3.2075899, 0.5272764,
		-4.8115144, 0, 0.7291982, -1.7486663, 1.5015489, -3.4289308, 2.1702385, 0, 0, -1.5236974, -2.2442245,
		-2.7002132, 2.6916146, 8.462519, -0.86479515, -2.6845582, 3.210464, -0.39838108, -5.1278195, -2.0957973,
		-3.574751, -0.80347896, 1.9967377, -0.4888832, 1.271344, -0.7196399, 4.1343956, 0.520421, 2.4047248, 0.48569518,
		1.2861887, -0.01585567},
	)
}
