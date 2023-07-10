package leaves

import (
	"bufio"
	"fmt"
	"github.com/dmitryikh/leaves/internal/xgjson"
	"os"

	"github.com/dmitryikh/leaves/internal/xgbin"
	"github.com/dmitryikh/leaves/transformation"
)

// XGBLinearFromReader reads  XGBoost's 'gblinear' model from `reader`
func XGBLinearFromReader(reader *bufio.Reader, loadTransformation bool) (*Ensemble, error) {
	e := &xgLinear{}

	//To support version after 1.0.0
	xgbin.ReadBinf(reader)
	// reading header info
	header, err := xgbin.ReadModelHeader(reader)
	if err != nil {
		return nil, err
	}
	gbLinearModel, err := xgbin.ReadGBLinearModel(reader)
	if err != nil {
		return nil, err
	}

	if header.NameGbm != "gblinear" {
		return nil, fmt.Errorf("only gblinear is supported (got %s). Use XGEnsembleFrom.. for gbtree", header.NameGbm)
	}
	if header.Param.NumFeatures == 0 {
		return nil, fmt.Errorf("zero number of features")
	}
	e.BaseScore = 0
	e.nRawOutputGroups = 1
	if header.Param.MajorVersion > uint32(0) {
		e.nRawOutputGroups = getNRawOutputGroups(header.Param.NumClass)
		e.BaseScore = calculateBaseScoreFromLearnerParam(float64(header.Param.BaseScore))
		e.NumFeature = int(header.Param.NumFeatures)
	} else {
		e.nRawOutputGroups = getNRawOutputGroups(gbLinearModel.Param.NumOutputGroup)
		e.BaseScore = float64(header.Param.BaseScore)
		e.NumFeature = int(gbLinearModel.Param.NumFeature)
	}
	e.Weights = gbLinearModel.Weights

	var transform transformation.Transform
	transform = &transformation.TransformRaw{NumOutputGroups: e.nRawOutputGroups}
	if loadTransformation {
		if header.NameObj == "binary:logistic" {
			transform = &transformation.TransformLogistic{}
		} else {
			return nil, fmt.Errorf("unknown transformation function '%s'", header.NameObj)
		}
	}
	return &Ensemble{e, transform}, nil
}

// XGBLinearFromFile reads XGBoost's 'gblinear' model from binary file
func XGBLinearFromFile(filename string, loadTransformation bool) (*Ensemble, error) {
	if ensemble, err := xgbLinearFromJson(filename, loadTransformation); err == nil {
		return ensemble, nil
	}
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return XGBLinearFromReader(bufReader, loadTransformation)
}

func xgbLinearFromJson(filename string, loadTransformation bool) (*Ensemble, error) {
	gbLinearJson, err := xgjson.ReadGBLinear(filename)
	if err != nil {
		return nil, err
	}
	e := &xgLinear{}
	gbLinearModel := gbLinearJson.Learner.GradientBooster.Model
	e.nRawOutputGroups = getNRawOutputGroups(gbLinearJson.Learner.LearnerModelParam.NumClass)
	e.NumFeature = int(gbLinearJson.Learner.LearnerModelParam.NumFeatures)
	e.Weights = gbLinearModel.Weights
	e.BaseScore = calculateBaseScoreFromLearnerParam(float64(gbLinearJson.Learner.LearnerModelParam.BaseScore))
	var transform transformation.Transform
	transform = &transformation.TransformRaw{NumOutputGroups: e.nRawOutputGroups}
	if loadTransformation {
		if gbLinearJson.Learner.Objective.Name == "binary:logistic" {
			transform = &transformation.TransformLogistic{}
		} else {
			return nil, fmt.Errorf("unknown transformation function '%s'", gbLinearJson.Learner.Objective.Name)
		}
	}
	return &Ensemble{e, transform}, nil
}
