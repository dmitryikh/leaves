package xgjson

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
)

func ReadGBTree(filePath string) (*GBTreeJson, error) {
	bytes, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	gbTree := &GBTreeJson{}
	if err := json.Unmarshal(bytes, gbTree); err != nil {
		return nil, err
	}
	if gbTree.Learner.GradientBooster.Name != "gbtree" && gbTree.Learner.GradientBooster.Name != "dart"{
		return nil, fmt.Errorf("wrong gbtree format, this reader can only read gbtree or dart")
	}
	return gbTree, nil
}

func ReadGBLinear(filePath string) (*GBLinearJson, error) {
	bytes, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	gbLinear := &GBLinearJson{}
	if err := json.Unmarshal(bytes, gbLinear); err != nil {
		return nil, err
	}
	if gbLinear.Learner.GradientBooster.Name != "gblinear" {
		return nil, fmt.Errorf("wrong gblinear format, this reader can only read gblinear")
	}
	return gbLinear, nil
}
