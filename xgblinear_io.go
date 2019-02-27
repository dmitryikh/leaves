package leaves

import (
	"bufio"
	"fmt"
	"os"

	"github.com/dmitryikh/leaves/internal/xgbin"
	"github.com/dmitryikh/leaves/transformation"
)

// XGBLinearFromReader reads  XGBoost's 'gblinear' model from `reader`
func XGBLinearFromReader(reader *bufio.Reader, loadTransformation bool) (*Ensemble, error) {
	e := &xgLinear{}

	// reading header info
	header, err := xgbin.ReadModelHeader(reader)
	if err != nil {
		return nil, err
	}
	if header.NameGbm != "gblinear" {
		return nil, fmt.Errorf("only gblinear is supported (got %s). Use XGEnsembleFrom.. for gbtree", header.NameGbm)
	}
	if header.Param.NumFeatures == 0 {
		return nil, fmt.Errorf("zero number of features")
	}
	e.BaseScore = float64(header.Param.BaseScore)

	gbLinearModel, err := xgbin.ReadGBLinearModel(reader)
	if err != nil {
		return nil, err
	}

	e.nRawOutputGroups = int(gbLinearModel.Param.NumOutputGroup)
	e.NumFeature = int(gbLinearModel.Param.NumFeature)
	e.Weights = gbLinearModel.Weights

	var transform transformation.Transform
	transform = &transformation.TransformRaw{e.nRawOutputGroups}
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
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return XGBLinearFromReader(bufReader, loadTransformation)
}
