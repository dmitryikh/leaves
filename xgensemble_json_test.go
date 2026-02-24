package leaves

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

const xgjsonTestdataDir = "internal/xgjson/testdata/"

// --- shared test helpers ---

type xgjsonExpected struct {
	Features    []float64 `json:"features"`
	RawScore    float64   `json:"raw_score"`
	Probability float64   `json:"probability"`
}

type xgjsonMultiExpected struct {
	Features      []float64   `json:"features"`
	RawScores     []float64   `json:"raw_scores"`
	Probabilities []float64   `json:"probabilities"`
}

type xgjsonPoissonExpected struct {
	Features   []float64 `json:"features"`
	RawScore   float64   `json:"raw_score"`
	Prediction float64   `json:"prediction"`
}

func loadXGJSONExpected(t *testing.T, path string) []xgjsonExpected {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read expected file: %v", err)
	}
	var out []xgjsonExpected
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal expected: %v", err)
	}
	return out
}

func loadXGJSONMultiExpected(t *testing.T, path string) []xgjsonMultiExpected {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read expected file: %v", err)
	}
	var out []xgjsonMultiExpected
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal expected: %v", err)
	}
	return out
}

func loadXGJSONPoissonExpected(t *testing.T, path string) []xgjsonPoissonExpected {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read expected file: %v", err)
	}
	var out []xgjsonPoissonExpected
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal expected: %v", err)
	}
	return out
}

func assertPredClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %.8f, want %.8f (diff %.2e > tol %.2e)", label, got, want, math.Abs(got-want), tol)
	}
}

// --- binary:logistic tests ---

func TestXGEnsembleJSON_BinaryLogistic(t *testing.T) {
	ensemble, err := XGEnsembleFromJSONFile(xgjsonTestdataDir+"test_binary_logistic.json", false)
	if err != nil {
		t.Fatalf("XGEnsembleFromJSONFile: %v", err)
	}

	if ensemble.NFeatures() != 3 {
		t.Errorf("NFeatures: got %d, want 3", ensemble.NFeatures())
	}
	if ensemble.NEstimators() != 10 {
		t.Errorf("NEstimators: got %d, want 10", ensemble.NEstimators())
	}
	if ensemble.NRawOutputGroups() != 1 {
		t.Errorf("NRawOutputGroups: got %d, want 1", ensemble.NRawOutputGroups())
	}

	expected := loadXGJSONExpected(t, xgjsonTestdataDir+"test_binary_logistic_expected.json")
	for _, e := range expected {
		got := ensemble.PredictSingle(e.Features, 0)
		assertPredClose(t, "json raw prediction", got, e.RawScore, 1e-5)
	}
}

func TestXGEnsembleUBJ_BinaryLogistic(t *testing.T) {
	ensemble, err := XGEnsembleFromUBJFile(xgjsonTestdataDir+"test_binary_logistic.ubj", false)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJFile: %v", err)
	}

	if ensemble.NFeatures() != 3 {
		t.Errorf("NFeatures: got %d, want 3", ensemble.NFeatures())
	}
	if ensemble.NEstimators() != 10 {
		t.Errorf("NEstimators: got %d, want 10", ensemble.NEstimators())
	}

	expected := loadXGJSONExpected(t, xgjsonTestdataDir+"test_binary_logistic_expected.json")
	for _, e := range expected {
		got := ensemble.PredictSingle(e.Features, 0)
		assertPredClose(t, "ubj raw prediction", got, e.RawScore, 1e-5)
	}
}

func TestXGEnsembleJSONEqualsUBJ_BinaryLogistic(t *testing.T) {
	jsonEnsemble, err := XGEnsembleFromJSONFile(xgjsonTestdataDir+"test_binary_logistic.json", false)
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	ubjEnsemble, err := XGEnsembleFromUBJFile(xgjsonTestdataDir+"test_binary_logistic.ubj", false)
	if err != nil {
		t.Fatalf("UBJ: %v", err)
	}

	expected := loadXGJSONExpected(t, xgjsonTestdataDir+"test_binary_logistic_expected.json")
	for _, e := range expected {
		jsonPred := jsonEnsemble.PredictSingle(e.Features, 0)
		ubjPred := ubjEnsemble.PredictSingle(e.Features, 0)
		// UBJ stores leaf values as float32; JSON uses float64 — allow for rounding
		assertPredClose(t, "json==ubj", jsonPred, ubjPred, 1e-5)
	}
}

func TestXGEnsembleFileDispatch(t *testing.T) {
	jsonEnsemble, err := XGEnsembleFromFile(xgjsonTestdataDir+"test_binary_logistic.json", false)
	if err != nil {
		t.Fatalf("XGEnsembleFromFile .json: %v", err)
	}
	ubjEnsemble, err := XGEnsembleFromFile(xgjsonTestdataDir+"test_binary_logistic.ubj", false)
	if err != nil {
		t.Fatalf("XGEnsembleFromFile .ubj: %v", err)
	}

	expected := loadXGJSONExpected(t, xgjsonTestdataDir+"test_binary_logistic_expected.json")
	for _, e := range expected {
		j := jsonEnsemble.PredictSingle(e.Features, 0)
		u := ubjEnsemble.PredictSingle(e.Features, 0)
		assertPredClose(t, "dispatch json==ubj", j, u, 1e-5)
	}
}

func TestXGEnsembleJSON_BinaryLogisticTransformation(t *testing.T) {
	ensemble, err := XGEnsembleFromJSONFile(xgjsonTestdataDir+"test_binary_logistic.json", true)
	if err != nil {
		t.Fatalf("XGEnsembleFromJSONFile with transform: %v", err)
	}

	if ensemble.NOutputGroups() != 1 {
		t.Errorf("NOutputGroups: got %d, want 1", ensemble.NOutputGroups())
	}

	expected := loadXGJSONExpected(t, xgjsonTestdataDir+"test_binary_logistic_expected.json")
	for _, e := range expected {
		got := ensemble.PredictSingle(e.Features, 0)
		assertPredClose(t, "sigmoid prediction", got, e.Probability, 1e-5)
	}
}

// --- reg:squarederror tests ---

func TestXGEnsembleJSON_Regression(t *testing.T) {
	ensemble, err := XGEnsembleFromJSONFile(xgjsonTestdataDir+"test_regression.json", false)
	if err != nil {
		t.Fatalf("XGEnsembleFromJSONFile: %v", err)
	}

	if ensemble.NFeatures() != 4 {
		t.Errorf("NFeatures: got %d, want 4", ensemble.NFeatures())
	}
	if ensemble.NEstimators() != 10 {
		t.Errorf("NEstimators: got %d, want 10", ensemble.NEstimators())
	}
	if ensemble.NRawOutputGroups() != 1 {
		t.Errorf("NRawOutputGroups: got %d, want 1", ensemble.NRawOutputGroups())
	}

	// reg:squarederror uses xgjsonExpected with raw_score only
	data, err := os.ReadFile(xgjsonTestdataDir + "test_regression_expected.json")
	if err != nil {
		t.Fatalf("read regression expected: %v", err)
	}
	var expected []struct {
		Features []float64 `json:"features"`
		RawScore float64   `json:"raw_score"`
	}
	if err := json.Unmarshal(data, &expected); err != nil {
		t.Fatalf("unmarshal regression expected: %v", err)
	}

	for _, e := range expected {
		got := ensemble.PredictSingle(e.Features, 0)
		assertPredClose(t, "regression raw", got, e.RawScore, 1e-5)
	}
}

// --- multi:softprob tests ---

func TestXGEnsembleJSON_Multiclass(t *testing.T) {
	expected := loadXGJSONMultiExpected(t, xgjsonTestdataDir+"test_multiclass_expected.json")

	t.Run("raw scores", func(t *testing.T) {
		ensemble, err := XGEnsembleFromJSONFile(xgjsonTestdataDir+"test_multiclass.json", false)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if ensemble.NFeatures() != 4 {
			t.Errorf("NFeatures: got %d, want 4", ensemble.NFeatures())
		}
		if ensemble.NEstimators() != 10 {
			t.Errorf("NEstimators: got %d, want 10", ensemble.NEstimators())
		}
		if ensemble.NRawOutputGroups() != 3 {
			t.Errorf("NRawOutputGroups: got %d, want 3", ensemble.NRawOutputGroups())
		}

		preds := make([]float64, ensemble.NOutputGroups())
		for _, e := range expected {
			if err := ensemble.Predict(e.Features, 0, preds); err != nil {
				t.Fatalf("Predict: %v", err)
			}
			for cls := 0; cls < 3; cls++ {
				assertPredClose(t, "multiclass raw", preds[cls], e.RawScores[cls], 1e-5)
			}
		}
	})

	t.Run("softmax probabilities", func(t *testing.T) {
		ensemble, err := XGEnsembleFromJSONFile(xgjsonTestdataDir+"test_multiclass.json", true)
		if err != nil {
			t.Fatalf("load with transform: %v", err)
		}
		if ensemble.NOutputGroups() != 3 {
			t.Errorf("NOutputGroups: got %d, want 3", ensemble.NOutputGroups())
		}

		preds := make([]float64, ensemble.NOutputGroups())
		for _, e := range expected {
			if err := ensemble.Predict(e.Features, 0, preds); err != nil {
				t.Fatalf("Predict: %v", err)
			}
			for cls := 0; cls < 3; cls++ {
				assertPredClose(t, "multiclass prob", preds[cls], e.Probabilities[cls], 1e-5)
			}
		}
	})

	t.Run("json equals ubj", func(t *testing.T) {
		jsonEnsemble, err := XGEnsembleFromJSONFile(xgjsonTestdataDir+"test_multiclass.json", false)
		if err != nil {
			t.Fatalf("JSON: %v", err)
		}
		ubjEnsemble, err := XGEnsembleFromUBJFile(xgjsonTestdataDir+"test_multiclass.ubj", false)
		if err != nil {
			t.Fatalf("UBJ: %v", err)
		}

		jsonPreds := make([]float64, jsonEnsemble.NOutputGroups())
		ubjPreds := make([]float64, ubjEnsemble.NOutputGroups())
		for _, e := range expected {
			if err := jsonEnsemble.Predict(e.Features, 0, jsonPreds); err != nil {
				t.Fatalf("JSON Predict: %v", err)
			}
			if err := ubjEnsemble.Predict(e.Features, 0, ubjPreds); err != nil {
				t.Fatalf("UBJ Predict: %v", err)
			}
			for cls := 0; cls < 3; cls++ {
				assertPredClose(t, "multiclass json==ubj", jsonPreds[cls], ubjPreds[cls], 1e-5)
			}
		}
	})
}

// --- count:poisson tests ---

func TestXGEnsembleJSON_Poisson(t *testing.T) {
	expected := loadXGJSONPoissonExpected(t, xgjsonTestdataDir+"test_poisson_expected.json")

	t.Run("raw margins", func(t *testing.T) {
		ensemble, err := XGEnsembleFromJSONFile(xgjsonTestdataDir+"test_poisson.json", false)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		for _, e := range expected {
			got := ensemble.PredictSingle(e.Features, 0)
			assertPredClose(t, "poisson raw", got, e.RawScore, 1e-5)
		}
	})

	t.Run("exp predictions", func(t *testing.T) {
		ensemble, err := XGEnsembleFromJSONFile(xgjsonTestdataDir+"test_poisson.json", true)
		if err != nil {
			t.Fatalf("load with transform: %v", err)
		}
		for _, e := range expected {
			got := ensemble.PredictSingle(e.Features, 0)
			assertPredClose(t, "poisson exp", got, e.Prediction, 1e-5)
		}
	})
}

