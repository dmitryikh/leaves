package transformation

type TransformLeafIndex struct {
	NumOutputGroups int
}

func (t *TransformLeafIndex) Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	// LeafIndex tranformation is treated in special way. Raw predictions is
	// already filled with leaf indices (but in float64 for compatibiliy)
	for i, v := range rawPredictions {
		outputPredictions[startIndex+i] = v
	}
	return nil
}

func (t *TransformLeafIndex) NOutputGroups() int {
	return t.NumOutputGroups
}

func (t *TransformLeafIndex) Type() TransformType {
	return LeafIndex
}

func (t *TransformLeafIndex) Name() string {
	return LeafIndex.Name()
}
