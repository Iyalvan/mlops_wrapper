# mlops-wrapper
custom python wrapper around mlflow

a unified mlflow wrapper that standardizes experiment tracking, artifact logging, and model registration across teams and projects. by providing decorators and base classes, this wrapper reduces boilerplate code and enforces best practices for mlflow usage.

## features

- **consistent logging**: standardized interface for logging parameters, metrics, and artifacts.
- **model registry integration**: automatically register models in mlflow’s model registry.
- **decorator & base class**: use a python decorator (`@mlflow_experiment`) or extend `baseexperiment` to minimize boilerplate code.
- **error handling**: automatically logs exceptions and ensures mlflow runs are closed gracefully.
- **flexible storage**: works with local or remote mlflow servers; supports s3 or local file system artifact stores.


## usage

do the following:

install the package:

```bash
pip install mlops-wrapper
```

then import it  `mlops_wrapper.mlflow_wrapper import mlflow_experiment` 

then decorate your training function with the `@mlflow_experiment` decorator

```python

@mlflow_experiment("MyExperiment")
def train_model(epochs=5):
    ## your training code
    return {"params": {...}, "metrics": {...}}

```

yet to be written