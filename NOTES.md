## 16.03.2019

Transformation functions are introduced. Before this step `leaves` was able to output only raw predictions. Here is new bool option named `loadTransformation` adedd to all model load functions: `XGEnsembleFromReader`, `XGEnsembleFromFile`, `XGBLinearFromReader`, `XGBLinearFromFile`, `SKEnsembleFromReader`, `SKEnsembleFromFile`, `LGEnsembleFromJSON`, `LGEnsembleFromReader`, `LGEnsembleFromFile`.

For example, line:
```go
model, err := leaves.LGEnsembleFromFile("lg_breast_cancer.model")
```

Should be changed to:
```go
model, err := leaves.LGEnsembleFromFile("lg_breast_cancer.model", false)
```

if one wants to leave old behaviour.


Also, `NClasses` `Ensemble` method will be renamed to `NRawOutputGroups` while keeping the same meaning - number of values that model provides for every object in raw predictions. There is also added `NOutputGroups` - number of values that model provides for every object after applying transformation function. Generally, that means that transformation function can change outputs dimensionality. Please note, if current transformation funciton is `raw`:

```go
model.Transformation().Name() == "raw"
```

then

```go
model.RawOutputGroups() == model.NOutputGroups()
```
