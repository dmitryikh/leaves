package leaves

import (
	"bufio"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/dmitryikh/leaves/util"
)

func TestReadLGTree(t *testing.T) {
	path := filepath.Join("testdata", "model_simple.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	// Read ensemble header (to skip)
	_, err = util.ReadParamsUntilBlank(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	// Read first tree only
	tree, err := lgTreeFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	if len(tree.nodes) != 2 {
		t.Fatalf("tree.nodes != 2 (got %d)", len(tree.nodes))
	}
	if tree.nCategorical != 1 {
		t.Fatalf("tree.nCategorical != 1 (got %d)", tree.nCategorical)
	}
	trueLeavesValues := []float64{0.56697267424823339, 0.3584987837673016, 0.41213915936587919}
	if err := util.AlmostEqualFloat64Slices(tree.leafValues, trueLeavesValues, 1e-10); err != nil {
		t.Fatalf("tree.leavesValues incorrect: %s", err.Error())
	}
	if tree.nodes[0].Flags&categorical == 0 {
		t.Fatal("first node should have categorical threshold")
	}
	if tree.nodes[0].Flags&catOneHot == 0 {
		t.Fatal("first node should have one hot decision rule")
	}
	if tree.nodes[0].Flags&leftLeaf == 0 {
		t.Fatal("first node should have right leaf")
	}
	if tree.nodes[0].Left != 0 {
		t.Fatal("first node should have leaf index 0")
	}
	if tree.nodes[0].Flags&missingNan == 0 {
		t.Fatal("first node should have missing nan")
	}
	if uint32(tree.nodes[0].Threshold) != 100 {
		t.Fatal("first node should have threshold = 100")
	}
	if tree.nodes[1].Flags&categorical != 0 {
		t.Fatal("second node should have numerical threshold")
	}
	if tree.nodes[1].Flags&defaultLeft == 0 {
		t.Fatal("second node should have default left")
	}
	if tree.nodes[1].Flags&rightLeaf == 0 {
		t.Fatal("second node should have left leaf")
	}
	if tree.nodes[1].Right != 2 {
		t.Fatal("second node should have leaf index 2")
	}
	if tree.nodes[1].Flags&leftLeaf == 0 {
		t.Fatal("second node should have right leaf")
	}
	if tree.nodes[1].Left != 1 {
		t.Fatal("second node should have leaf index 1")
	}
}

func TestLGTreeLeaf1(t *testing.T) {
	path := filepath.Join("testdata", "tree_1leaf.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	tree, err := lgTreeFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	if tree.nLeaves() != 1 {
		t.Fatalf("expected tree with 1 leaves (got %d)", tree.nLeaves())
	}
	if tree.nNodes() != 0 {
		t.Fatalf("expected tree with 0 node (got %d)", tree.nNodes())
	}

	fvals := []float64{0.0}
	check := func(truePred float64) {
		p := tree.predict(fvals)
		if !util.AlmostEqualFloat64(p, truePred, 1e-3) {
			t.Errorf("expected prediction %f, got %f", truePred, p)
		}
	}

	check(0.123)
	fvals[0] = 10.0
	check(0.123)
	fvals[0] = -10.0
	check(0.123)
	fvals[0] = math.NaN()
	check(0.123)
}

func TestLGTreeLeaves2(t *testing.T) {
	path := filepath.Join("testdata", "tree_2leaves.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	tree, err := lgTreeFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	if tree.nLeaves() != 2 {
		t.Fatalf("expected tree with 2 leaves (got %d)", tree.nLeaves())
	}
	if tree.nNodes() != 1 {
		t.Fatalf("expected tree with 1 node (got %d)", tree.nNodes())
	}

	fvals := []float64{0.0}
	check := func(truePred float64) {
		p := tree.predict(fvals)
		if !util.AlmostEqualFloat64(p, truePred, 1e-3) {
			t.Errorf("expected prediction %f, got %f", truePred, p)
		}
	}

	check(0.43)
	fvals[0] = 5.1
	check(0.59)
	fvals[0] = math.NaN()
	check(0.43)
}

func TestLGTreeLeaves3(t *testing.T) {
	path := filepath.Join("testdata", "tree_3leaves.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	tree, err := lgTreeFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	if tree.nLeaves() != 3 {
		t.Fatalf("expected tree with 3 leaves (got %d)", tree.nLeaves())
	}
	if tree.nNodes() != 2 {
		t.Fatalf("expected tree with 2 node (got %d)", tree.nNodes())
	}

	fvals := []float64{0.0, 0.0}
	check := func(truePred float64) {
		p := tree.predict(fvals)
		if !util.AlmostEqualFloat64(p, truePred, 1e-3) {
			t.Errorf("expected prediction %f, got %f", truePred, p)
		}
	}

	check(0.35)
	fvals[0] = 1000.0
	check(0.38)
	fvals[0] = math.NaN()
	check(0.35)
	fvals[1] = 10.0
	check(0.35)
	fvals[1] = 100.0
	check(0.54)
}

func TestLGEnsemble(t *testing.T) {
	path := filepath.Join("testdata", "model_simple.txt")
	model, err := LGEnsembleFromFile(path, false)
	if err != nil {
		t.Fatal(err)
	}
	if model.NEstimators() != 2 {
		t.Fatalf("expected 2 trees (got %d)", model.NEstimators())
	}

	denseValues := []float64{0.0, 0.0,
		1000.0, 0.0,
		800.0, 0.0,
		800.0, 100,
		0.0, 100,
		1000, math.NaN(),
		math.NaN(), math.NaN(),
	}

	denseRows := 7
	denseCols := 2

	// check predictions
	predictions := make([]float64, denseRows)
	model.PredictDense(denseValues, denseRows, denseCols, predictions, 0, 0)
	truePredictions := []float64{0.29462594, 0.39565483, 0.39565483, 0.69580371, 0.69580371, 0.39565483, 0.29462594}
	if err := util.AlmostEqualFloat64Slices(predictions, truePredictions, 1e-7); err != nil {
		t.Fatalf("predictions on dense not correct (all trees): %s", err.Error())
	}

	// check prediction only on first tree
	model.PredictDense(denseValues, denseRows, denseCols, predictions, 1, 0)
	truePredictions = []float64{0.35849878, 0.41213916, 0.41213916, 0.56697267, 0.56697267, 0.41213916, 0.35849878}
	if err := util.AlmostEqualFloat64Slices(predictions, truePredictions, 1e-7); err != nil {
		t.Fatalf("predictions on dense not correct (all trees): %s", err.Error())
	}
}

func TestLGEnsembleJSON1tree1leaf(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_1tree_1leaf.json")
	// loading model
	modelFile, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer modelFile.Close()
	model, err := LGEnsembleFromJSON(modelFile, false)
	if err != nil {
		t.Fatal(err)
	}
	if model.NEstimators() != 1 {
		t.Fatalf("expected 1 trees (got %d)", model.NEstimators())
	}

	if model.NOutputGroups() != 1 {
		t.Fatalf("expected 1 class (got %d)", model.NOutputGroups())
	}

	if model.NFeatures() != 41 {
		t.Fatalf("expected 41 class (got %d)", model.NFeatures())
	}

	features := make([]float64, model.NFeatures())
	pred := model.PredictSingle(features, 0)
	if pred != 0.42 {
		t.Fatalf("expected prediction 0.42 (got %f)", pred)
	}
}

func TestLGEnsembleJSON1tree(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_1tree.json")
	// loading model
	modelFile, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer modelFile.Close()
	model, err := LGEnsembleFromJSON(modelFile, false)
	if err != nil {
		t.Fatal(err)
	}
	if model.NEstimators() != 1 {
		t.Fatalf("expected 1 trees (got %d)", model.NEstimators())
	}

	if model.NOutputGroups() != 1 {
		t.Fatalf("expected 1 class (got %d)", model.NOutputGroups())
	}

	if model.NFeatures() != 2 {
		t.Fatalf("expected 2 class (got %d)", model.NFeatures())
	}

	check := func(features []float64, trueAnswer float64) {
		pred := model.PredictSingle(features, 0)
		if pred != trueAnswer {
			t.Fatalf("expected prediction %f (got %f)", trueAnswer, pred)
		}
	}

	check([]float64{0.0, 0.0}, 0.4242)
	check([]float64{0.0, 11.0}, 0.4242)
	check([]float64{0.13, 11.0}, 0.4242)
	check([]float64{0.0, 1.0}, 0.4703)
	check([]float64{0.0, 10.0}, 0.4703)
	check([]float64{0.0, 100.0}, 0.4703)
	check([]float64{0.15, 0.0}, 1.1111)
	check([]float64{0.15, 11.0}, 1.1111)
}
