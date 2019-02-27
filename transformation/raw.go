package transformation

type TransformRaw struct {
	NumOutputGroups int
}

func (t *TransformRaw) Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	for i, v := range rawPredictions {
		outputPredictions[startIndex+i] = v
	}
	return nil
}

func (t *TransformRaw) NOutputGroups() int {
	return t.NumOutputGroups
}

func (t *TransformRaw) Type() TransformType {
	return Raw
}

func (t *TransformRaw) Name() string {
	return Raw.Name()
}
