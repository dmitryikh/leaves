package leaves

import (
	"bufio"
	"fmt"
	"os"

	"github.com/dmitryikh/leaves/internal/xgbin"
)

// XGBLinearFromReader reads  XGBoost's 'gblinear' model from `reader`
func XGBLinearFromReader(reader *bufio.Reader, loadTransformation bool) (*Ensemble, error) {
	e := &xgLinear{}

	if loadTransformation {
		return nil, fmt.Errorf("transformation functions are not supported for XGBoost models")
	}

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
	e.nClasses = int(gbLinearModel.Param.NumOutputGroup)
	e.NumFeature = int(gbLinearModel.Param.NumFeature)
	e.Weights = gbLinearModel.Weights
	return &Ensemble{e}, nil
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
