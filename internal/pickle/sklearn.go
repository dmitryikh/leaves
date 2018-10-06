package pickle

import (
	"encoding/binary"
	"fmt"

	"github.com/dmitryikh/leaves/util"
)

// SklearnNode represents tree node data structure
type SklearnNode struct {
	LeftChild            int
	RightChild           int
	Feature              int
	Threshold            float64
	Impurity             float64
	NNodeSamples         int
	WeightedNNodeSamples float64
}

// SklearnNodeFromBytes converts 56 raw bytes into SklearnNode struct
// The rule described in https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx#L70 (NODE_DTYPE)
// 'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples'],
// 'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp, np.float64]
func SklearnNodeFromBytes(bytes []byte) SklearnNode {
	offset := 0
	size := 8
	node := SklearnNode{}
	node.LeftChild = int(binary.LittleEndian.Uint64(bytes[offset : offset+size]))
	offset += size
	node.RightChild = int(binary.LittleEndian.Uint64(bytes[offset : offset+size]))
	offset += size
	node.Feature = int(binary.LittleEndian.Uint64(bytes[offset : offset+size]))
	offset += size
	node.Threshold = util.Float64FromBytes(bytes[offset:offset+size], true)
	offset += size
	node.Impurity = util.Float64FromBytes(bytes[offset:offset+size], true)
	offset += size
	node.NNodeSamples = int(binary.LittleEndian.Uint64(bytes[offset : offset+size]))
	offset += size
	node.WeightedNNodeSamples = util.Float64FromBytes(bytes[offset:offset+size], true)
	return node
}

// SklearnTree represents parsed tree struct
type SklearnTree struct {
	NOutputs int
	Classes  []int
	NNodes   int
	Nodes    []SklearnNode
	Values   []float64
}

func (t *SklearnTree) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "sklearn.tree._tree", "Tree")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("expected len(reduce.Args) = 3 (got %d)", len(reduce.Args))
	}

	var ok bool
	t.NOutputs, ok = reduce.Args[2].(int)
	if !ok {
		return fmt.Errorf("expected int (got %T)", reduce.Args[2])
	}

	arr := NumpyArrayRaw{}
	err = ParseClass(&arr, reduce.Args[1])
	if err != nil {
		return err
	}

	if len(arr.Shape) != 1 && arr.Shape[0] != t.NOutputs {
		return fmt.Errorf("expected 1 dim array with %d values (got: %v)", t.NOutputs, arr.Shape)
	}
	if arr.Type.Type != "i8" || arr.Type.LittleEndinan != true {
		return fmt.Errorf("expected ndtype \"i8\" little endian (got: %#v)", arr.Type)
	}

	t.Classes = make([]int, 0, t.NOutputs)
	err = arr.Data.Iterate(8, func(b []byte) error {
		t.Classes = append(t.Classes, int(binary.LittleEndian.Uint64(b)))
		return nil
	})
	if err != nil {
		return
	}
	return
}

func (t *SklearnTree) Build(build Build) (err error) {
	dict, err := toDict(build.Args)
	if err != nil {
		return
	}
	t.NNodes, err = dict.toInt("node_count")
	if err != nil {
		return
	}
	nodesObj, err := dict.value("nodes")
	if err != nil {
		return
	}
	arr := NumpyArrayRaw{}
	err = ParseClass(&arr, nodesObj)
	if err != nil {
		return
	}
	if arr.Type.Type != "V56" {
		return fmt.Errorf("expected arr.Type.Type = \"V56\" (got: %s)", arr.Type.Type)
	}
	err = arr.Data.Iterate(56, func(b []byte) error {
		t.Nodes = append(t.Nodes, SklearnNodeFromBytes(b))
		return nil
	})
	if err != nil {
		return
	}
	valuesObj, err := dict.value("values")
	if err != nil {
		return
	}
	arr = NumpyArrayRaw{}
	err = ParseClass(&arr, valuesObj)
	if err != nil {
		return
	}
	if arr.Type.Type != "f8" {
		return fmt.Errorf("expected ndtype \"f8\" (got: %#v)", arr.Type)
	}
	err = arr.Data.Iterate(8, func(b []byte) error {
		t.Values = append(t.Values, util.Float64FromBytes(b, arr.Type.LittleEndinan))
		return nil
	})
	if err != nil {
		return
	}
	return
}

type SklearnDecisionTreeRegressor struct {
	Tree        SklearnTree
	NClasses    int
	MaxFeatures int
	NOutputs    int
}

func (t *SklearnDecisionTreeRegressor) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "copy_reg", "_reconstructor")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	_, err = toGlobal(reduce.Args[0], "sklearn.tree.tree", "DecisionTreeRegressor")
	if err != nil {
		return
	}
	return
}

func (t *SklearnDecisionTreeRegressor) Build(build Build) (err error) {
	dict, err := toDict(build.Args)
	if err != nil {
		return
	}
	nClassesRaw := NumpyScalarRaw{}
	nClassesObj, err := dict.value("n_classes_")
	if err != nil {
		return
	}
	err = ParseClass(&nClassesRaw, nClassesObj)
	if err != nil {
		return
	}
	if nClassesRaw.Type.Type != "i8" || !nClassesRaw.Type.LittleEndinan {
		return fmt.Errorf("expected little endian i8, got (%#v)", nClassesRaw.Type)
	}
	t.NClasses = int(binary.LittleEndian.Uint64(nClassesRaw.Data))
	t.MaxFeatures, err = dict.toInt("max_features_")
	if err != nil {
		return
	}
	t.NOutputs, err = dict.toInt("n_outputs_")
	if err != nil {
		return
	}

	treeObj, err := dict.value("tree_")
	if err != nil {
		return
	}
	err = ParseClass(&t.Tree, treeObj)
	if err != nil {
		return
	}
	return
}

type SklearnGradientBoosting struct {
	NClasses      int
	Classes       []int
	NEstimators   int
	MaxFeatures   int
	Estimators    []SklearnDecisionTreeRegressor
	LearningRate  float64
	InitEstimator SKlearnInitEstimator
}

func (t *SklearnGradientBoosting) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "copy_reg", "_reconstructor")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	_, err = toGlobal(reduce.Args[0], "sklearn.ensemble.gradient_boosting", "GradientBoostingClassifier")
	if err != nil {
		return
	}
	return
}

func (t *SklearnGradientBoosting) Build(build Build) (err error) {
	dict, err := toDict(build.Args)
	if err != nil {
		return
	}
	t.NClasses, err = dict.toInt("n_classes_")
	if err != nil {
		return
	}

	t.LearningRate, err = dict.toFloat("learning_rate")
	if err != nil {
		return
	}

	obj, err := dict.value("loss")
	if err != nil {
		return
	}
	_ /* loss */, err = toUnicode(obj, -1)
	if err != nil {
		return
	}

	obj, err = dict.value("init_")
	if err != nil {
		return
	}
	err = ParseClass(&t.InitEstimator, obj)
	if err != nil {
		return
	}

	arr := NumpyArrayRaw{}
	classesObj, err := dict.value("classes_")
	if err != nil {
		return err
	}
	err = ParseClass(&arr, classesObj)
	if err != nil {
		return err
	}

	if len(arr.Shape) != 1 && arr.Shape[0] != t.NClasses {
		return fmt.Errorf("expected 1 dim array with %d values (got: %v)", t.NClasses, arr.Shape)
	}
	if arr.Type.Type != "i8" || arr.Type.LittleEndinan != true {
		return fmt.Errorf("expected ndtype \"i8\" little endian (got: %#v)", arr.Type)
	}

	t.Classes = make([]int, 0, t.NClasses)
	err = arr.Data.Iterate(8, func(b []byte) error {
		t.Classes = append(t.Classes, int(binary.LittleEndian.Uint64(b)))
		return nil
	})
	if err != nil {
		return
	}
	t.MaxFeatures, err = dict.toInt("max_features_")
	if err != nil {
		return
	}
	t.NEstimators, err = dict.toInt("n_estimators")
	if err != nil {
		return
	}
	//  estimators_ : ndarray of DecisionTreeRegressor,\
	//  shape (n_estimators, ``loss_.K``)
	//  		The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
	//  		classification, otherwise n_classes.
	arr = NumpyArrayRaw{}
	obj, err = dict.value("estimators_")
	if err != nil {
		return
	}
	err = ParseClass(&arr, obj)
	if err != nil {
		return
	}
	adjNClasses := t.NClasses
	if t.NClasses == 2 {
		adjNClasses = 1
	}
	if len(arr.Shape) != 2 || arr.Shape[0] != t.NEstimators || arr.Shape[1] != adjNClasses {
		return fmt.Errorf("unexpected shape: %#v", arr.Shape)
	}
	if len(arr.DataList) != arr.Shape[0]*arr.Shape[1] {
		return fmt.Errorf("unexpected array list length")
	}
	t.Estimators = make([]SklearnDecisionTreeRegressor, arr.Shape[0]*arr.Shape[1])
	for i := range arr.DataList {
		err = ParseClass(&t.Estimators[i], arr.DataList[i])
		if err != nil {
			return
		}
	}
	return
}

type SKlearnInitEstimator struct {
	Name  string
	Prior []float64
}

func (e *SKlearnInitEstimator) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "copy_reg", "_reconstructor")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	classDesc, err := toGlobal(reduce.Args[0], "sklearn.ensemble.gradient_boosting", "")
	if err != nil {
		return
	}
	e.Name = classDesc.Name
	return
}

func (e *SKlearnInitEstimator) Build(build Build) (err error) {
	if e.Name == "LogOddsEstimator" {
		dict, err := toDict(build.Args)
		if err != nil {
			return err
		}
		priorObj, err := dict.value("prior")
		if err != nil {
			return err
		}
		numpyScalar := NumpyScalarRaw{}
		err = ParseClass(&numpyScalar, priorObj)
		if err != nil {
			return err
		}
		if numpyScalar.Type.Type != "f8" {
			return fmt.Errorf("expected f8, got (%#v)", numpyScalar.Type)
		}
		e.Prior = append(e.Prior, util.Float64FromBytes(numpyScalar.Data, numpyScalar.Type.LittleEndinan))
	} else if e.Name == "PriorProbabilityEstimator" {
		dict, err := toDict(build.Args)
		if err != nil {
			return err
		}
		priorObj, err := dict.value("priors")
		if err != nil {
			return err
		}
		numpyArray := NumpyArrayRaw{}
		err = ParseClass(&numpyArray, priorObj)
		if err != nil {
			return err
		}
		if numpyArray.Type.Type != "f8" {
			return fmt.Errorf("expected f8, got (%#v)", numpyArray.Type)
		}
		numpyArray.Data.Iterate(8, func(bytes []byte) error {
			e.Prior = append(e.Prior, util.Float64FromBytes(bytes, numpyArray.Type.LittleEndinan))
			return nil
		})
	} else {
		return fmt.Errorf("unknown init estimator class: %s", e.Name)
	}
	return
}
