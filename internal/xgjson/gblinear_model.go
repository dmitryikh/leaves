package xgjson

import "github.com/dmitryikh/leaves/internal/xgbin"

type GBLinearJson struct {
	Learner GBLinearLearner `json:"learner"`
	Version []int           `json:"version"`
}

type GBLinearLearner struct {
	FeatureNames      []string                      `json:"feature_names"`
	FeatureTypes      []string                      `json:"feature_types"`
	GradientBooster   GBLinearBooster               `json:"gradient_booster"`
	Objective         Objective                     `json:"objective"`
	LearnerModelParam xgbin.LearnerModelParamLegacy `json:"learner_model_param"`
}

type GBLinearBooster struct {
	Model xgbin.GBLinearModel `json:"model"`
	Name  string              `json:"name"`
}