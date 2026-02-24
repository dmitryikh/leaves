// Package xgjson reads XGBoost models saved in JSON (.json) and Universal Binary JSON (.ubj) formats.
// These are the modern formats produced by XGBoost 2.x and 3.x.
//
// Use ReadModelJSON for .json files and ReadModelUBJ for .ubj files.
// Both return *ModelJSON with the same logical structure.
package xgjson

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/dmitryikh/leaves/internal/ubjdecode"
)

// ModelJSON is the top-level XGBoost model structure.
type ModelJSON struct {
	Learner LearnerJSON
	Version []int32
}

// LearnerJSON holds the learner sub-tree of the model.
type LearnerJSON struct {
	FeatureNames      []string
	GradientBooster   GradientBoosterJSON
	LearnerModelParam LearnerModelParamJSON
	Objective         ObjectiveJSON
}

// LearnerModelParamJSON holds global model parameters.
// Numeric fields are stored as strings (or bracketed strings like "[5.2E-1]") by XGBoost.
type LearnerModelParamJSON struct {
	BaseScore  string // may be "[value]" in XGBoost 2.x
	NumClass   string
	NumFeature string
}

// ObjectiveJSON holds the objective function name.
type ObjectiveJSON struct {
	Name string
}

// GradientBoosterJSON wraps the booster model and its name.
type GradientBoosterJSON struct {
	Model GBTreeModelDataJSON
	Name  string // "gbtree" or "dart"
}

// GBTreeModelDataJSON holds the tree ensemble data.
type GBTreeModelDataJSON struct {
	GBTreeModelParam GBTreeModelParamJSON
	TreeInfo         []int32
	Trees            []TreeJSON
}

// GBTreeModelParamJSON holds ensemble-level parameters.
type GBTreeModelParamJSON struct {
	NumParallelTree string
	NumTrees        string
}

// TreeJSON holds a single decision tree's data.
type TreeJSON struct {
	ID              int32
	BaseWeights     []float64
	DefaultLeft     []int32
	LeftChildren    []int32
	RightChildren   []int32
	SplitConditions []float64
	SplitIndices    []int32
	SplitType       []int32
	NumNodes        string // from tree_param.num_nodes
}

// ---------------------------------------------------------------------------
// JSON format (encoding/json with struct tags)
// ---------------------------------------------------------------------------

// jsonModel is the raw JSON-tagged equivalent used only for unmarshaling.
type jsonModel struct {
	Learner struct {
		FeatureNames    []string `json:"feature_names"`
		GradientBooster struct {
			Model struct {
				GBTreeModelParam struct {
					NumParallelTree string `json:"num_parallel_tree"`
					NumTrees        string `json:"num_trees"`
				} `json:"gbtree_model_param"`
				TreeInfo []int32 `json:"tree_info"`
				Trees    []struct {
					ID              int32     `json:"id"`
					BaseWeights     []float64 `json:"base_weights"`
					DefaultLeft     []int32   `json:"default_left"`
					LeftChildren    []int32   `json:"left_children"`
					RightChildren   []int32   `json:"right_children"`
					SplitConditions []float64 `json:"split_conditions"`
					SplitIndices    []int32   `json:"split_indices"`
					SplitType       []int32   `json:"split_type"`
					TreeParam       struct {
						NumNodes string `json:"num_nodes"`
					} `json:"tree_param"`
				} `json:"trees"`
			} `json:"model"`
			Name string `json:"name"`
		} `json:"gradient_booster"`
		LearnerModelParam struct {
			BaseScore  string `json:"base_score"`
			NumClass   string `json:"num_class"`
			NumFeature string `json:"num_feature"`
		} `json:"learner_model_param"`
		Objective struct {
			Name string `json:"name"`
		} `json:"objective"`
	} `json:"learner"`
	Version []int32 `json:"version"`
}

// ReadModelJSON reads an XGBoost model from a JSON reader.
func ReadModelJSON(r io.Reader) (*ModelJSON, error) {
	var raw jsonModel
	if err := json.NewDecoder(r).Decode(&raw); err != nil {
		return nil, fmt.Errorf("xgjson: JSON decode: %w", err)
	}
	return jsonModelToModelJSON(&raw), nil
}

func jsonModelToModelJSON(raw *jsonModel) *ModelJSON {
	m := &ModelJSON{
		Version: raw.Version,
	}
	rl := &raw.Learner
	m.Learner.FeatureNames = rl.FeatureNames
	m.Learner.LearnerModelParam = LearnerModelParamJSON{
		BaseScore:  rl.LearnerModelParam.BaseScore,
		NumClass:   rl.LearnerModelParam.NumClass,
		NumFeature: rl.LearnerModelParam.NumFeature,
	}
	m.Learner.Objective = ObjectiveJSON{Name: rl.Objective.Name}

	gb := &rl.GradientBooster
	m.Learner.GradientBooster = GradientBoosterJSON{
		Name: gb.Name,
		Model: GBTreeModelDataJSON{
			GBTreeModelParam: GBTreeModelParamJSON{
				NumParallelTree: gb.Model.GBTreeModelParam.NumParallelTree,
				NumTrees:        gb.Model.GBTreeModelParam.NumTrees,
			},
			TreeInfo: gb.Model.TreeInfo,
		},
	}

	for _, t := range gb.Model.Trees {
		m.Learner.GradientBooster.Model.Trees = append(
			m.Learner.GradientBooster.Model.Trees,
			TreeJSON{
				ID:              t.ID,
				BaseWeights:     t.BaseWeights,
				DefaultLeft:     t.DefaultLeft,
				LeftChildren:    t.LeftChildren,
				RightChildren:   t.RightChildren,
				SplitConditions: t.SplitConditions,
				SplitIndices:    t.SplitIndices,
				SplitType:       t.SplitType,
				NumNodes:        t.TreeParam.NumNodes,
			},
		)
	}
	return m
}

// ---------------------------------------------------------------------------
// UBJ format — walk map[string]interface{} from ubjdecode
// ---------------------------------------------------------------------------

// ReadModelUBJ reads an XGBoost model from a UBJ reader.
// Uses ubjdecode internally; produces the same ModelJSON struct.
func ReadModelUBJ(r io.Reader) (*ModelJSON, error) {
	raw, err := ubjdecode.DecodeValue(r)
	if err != nil {
		return nil, fmt.Errorf("xgjson: UBJ decode: %w", err)
	}
	root, ok := raw.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("xgjson: UBJ root is not an object")
	}
	return ubjMapToModelJSON(root)
}

// ---------------------------------------------------------------------------
// UBJ map-walking helpers
// ---------------------------------------------------------------------------

func ubjMapToModelJSON(root map[string]interface{}) (*ModelJSON, error) {
	m := &ModelJSON{}

	if v, ok := root["version"]; ok {
		m.Version = toInt32Slice(v)
	}

	learnerRaw, ok := root["learner"]
	if !ok {
		return nil, fmt.Errorf("xgjson: UBJ missing 'learner' key")
	}
	learnerMap, ok := learnerRaw.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("xgjson: UBJ 'learner' is not an object")
	}

	// feature_names
	if v, ok := learnerMap["feature_names"]; ok {
		m.Learner.FeatureNames = toStringSlice(v)
	}

	// learner_model_param
	if v, ok := learnerMap["learner_model_param"]; ok {
		if mp, ok := v.(map[string]interface{}); ok {
			m.Learner.LearnerModelParam = LearnerModelParamJSON{
				BaseScore:  mapString(mp, "base_score"),
				NumClass:   mapString(mp, "num_class"),
				NumFeature: mapString(mp, "num_feature"),
			}
		}
	}

	// objective
	if v, ok := learnerMap["objective"]; ok {
		if om, ok := v.(map[string]interface{}); ok {
			m.Learner.Objective = ObjectiveJSON{Name: mapString(om, "name")}
		}
	}

	// gradient_booster
	if v, ok := learnerMap["gradient_booster"]; ok {
		gbMap, ok := v.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("xgjson: UBJ 'gradient_booster' is not an object")
		}
		gb, err := ubjGradientBooster(gbMap)
		if err != nil {
			return nil, err
		}
		m.Learner.GradientBooster = *gb
	}

	return m, nil
}

func ubjGradientBooster(gbMap map[string]interface{}) (*GradientBoosterJSON, error) {
	gb := &GradientBoosterJSON{
		Name: mapString(gbMap, "name"),
	}

	modelRaw, ok := gbMap["model"]
	if !ok {
		return nil, fmt.Errorf("xgjson: UBJ gradient_booster missing 'model'")
	}
	modelMap, ok := modelRaw.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("xgjson: UBJ gradient_booster 'model' is not an object")
	}

	// gbtree_model_param
	if v, ok := modelMap["gbtree_model_param"]; ok {
		if pm, ok := v.(map[string]interface{}); ok {
			gb.Model.GBTreeModelParam = GBTreeModelParamJSON{
				NumParallelTree: mapString(pm, "num_parallel_tree"),
				NumTrees:        mapString(pm, "num_trees"),
			}
		}
	}

	// tree_info
	if v, ok := modelMap["tree_info"]; ok {
		gb.Model.TreeInfo = toInt32Slice(v)
	}

	// trees
	treesRaw, ok := modelMap["trees"]
	if !ok {
		return nil, fmt.Errorf("xgjson: UBJ model missing 'trees'")
	}
	treesSlice, ok := treesRaw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("xgjson: UBJ 'trees' is not an array")
	}
	for i, tRaw := range treesSlice {
		tMap, ok := tRaw.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("xgjson: UBJ tree %d is not an object", i)
		}
		tree, err := ubjTree(tMap)
		if err != nil {
			return nil, fmt.Errorf("xgjson: UBJ tree %d: %w", i, err)
		}
		gb.Model.Trees = append(gb.Model.Trees, *tree)
	}

	return gb, nil
}

func ubjTree(tMap map[string]interface{}) (*TreeJSON, error) {
	t := &TreeJSON{}

	if v, ok := tMap["id"]; ok {
		t.ID = int32(toInt64(v))
	}
	t.BaseWeights = toFloat64Slice(tMap["base_weights"])
	t.DefaultLeft = toInt32Slice(tMap["default_left"])
	t.LeftChildren = toInt32Slice(tMap["left_children"])
	t.RightChildren = toInt32Slice(tMap["right_children"])
	t.SplitConditions = toFloat64Slice(tMap["split_conditions"])
	t.SplitIndices = toInt32Slice(tMap["split_indices"])
	t.SplitType = toInt32Slice(tMap["split_type"])

	if v, ok := tMap["tree_param"]; ok {
		if pm, ok := v.(map[string]interface{}); ok {
			t.NumNodes = mapString(pm, "num_nodes")
		}
	}
	return t, nil
}

// ---------------------------------------------------------------------------
// Type coercion helpers
// ---------------------------------------------------------------------------

func mapString(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
		return fmt.Sprintf("%v", v)
	}
	return ""
}

func toInt64(v interface{}) int64 {
	switch x := v.(type) {
	case int64:
		return x
	case int32:
		return int64(x)
	case float64:
		return int64(x)
	case float32:
		return int64(x)
	default:
		return 0
	}
}

// toInt32Slice coerces any of the UBJ typed array forms to []int32.
func toInt32Slice(v interface{}) []int32 {
	if v == nil {
		return nil
	}
	switch x := v.(type) {
	case []int32:
		return x
	case []int64:
		out := make([]int32, len(x))
		for i, vv := range x {
			out[i] = int32(vv)
		}
		return out
	case []interface{}:
		out := make([]int32, len(x))
		for i, vv := range x {
			out[i] = int32(toInt64(vv))
		}
		return out
	default:
		return nil
	}
}

// toFloat64Slice coerces any of the UBJ typed array forms to []float64.
func toFloat64Slice(v interface{}) []float64 {
	if v == nil {
		return nil
	}
	switch x := v.(type) {
	case []float64:
		return x
	case []float32:
		out := make([]float64, len(x))
		for i, vv := range x {
			out[i] = float64(vv)
		}
		return out
	case []interface{}:
		out := make([]float64, len(x))
		for i, vv := range x {
			switch f := vv.(type) {
			case float64:
				out[i] = f
			case float32:
				out[i] = float64(f)
			case int64:
				out[i] = float64(f)
			}
		}
		return out
	default:
		return nil
	}
}

func toStringSlice(v interface{}) []string {
	if v == nil {
		return nil
	}
	switch x := v.(type) {
	case []string:
		return x
	case []interface{}:
		out := make([]string, len(x))
		for i, vv := range x {
			if s, ok := vv.(string); ok {
				out[i] = s
			} else {
				out[i] = fmt.Sprintf("%v", vv)
			}
		}
		return out
	default:
		return nil
	}
}

// ParseBaseScore parses XGBoost's base_score string, which may be wrapped in
// brackets like "[5.2313304E-1]" (XGBoost 2.x) or a plain float string.
// For multi-class models (XGBoost 3.x) with comma-separated per-class values,
// use ParseBaseScoreMulti instead.
func ParseBaseScore(s string) (float64, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")
	s = strings.TrimSpace(s)
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("xgjson: parse base_score %q: %w", s, err)
	}
	return v, nil
}

// ParseBaseScoreMulti parses XGBoost's base_score string, returning one float64
// per output group. Handles both single-value (all groups share same score) and
// comma-separated per-group strings like "[1.4E-2,-2.2E-2,8.1E-3]" from XGBoost 3.x
// multi-class models.
func ParseBaseScoreMulti(s string) ([]float64, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")
	s = strings.TrimSpace(s)
	parts := strings.Split(s, ",")
	out := make([]float64, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, fmt.Errorf("xgjson: parse base_score[%d] in %q: %w", i, s, err)
		}
		out[i] = v
	}
	return out, nil
}
