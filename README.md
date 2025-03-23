# mlops-wrapper
custom python wrapper around mlflow

a unified mlflow wrapper that standardizes experiment tracking, artifact logging, and model registration across teams and projects. by providing decorators and base classes, this wrapper reduces boilerplate code and enforces best practices for mlflow usage.


## why use this wrapper?

- **simplify mlflow usage**: no need to manually start/stop mlflow runs or remember to log parameters/metrics/artifacts each time.
- **built-in flexibility**: pass a single flag to enable autologging or choose to manually specify which details get logged.
- **extendable**: easily add custom tags or additional params by returning them in a dictionary from your decorated function.
- **consistent experiment tracking**: standardize how your team logs experiments, making it easier to compare runs and share best practices.

## features

- **consistent logging**: standardized interface for logging parameters, metrics, and artifacts.
- **model registry integration**: automatically register models in mlflow’s model registry.
- **decorator & base class**: use a python decorator (`@mlflow_experiment`) or extend `baseexperiment` to minimize boilerplate code.
- **error handling**: automatically logs exceptions and ensures mlflow runs are closed gracefully.


## quick start

do the following:

install the package:

```bash
pip install mlops-wrapper
```

then import it  `mlops_wrapper.mlflow_wrapper import mlflow_experiment` 

then decorate your training function with the `@mlflow_experiment` decorator

### for autologging

with autologging enabled, the mlflow will automatically log all parameters, metrics, and artifacts from your training function with its intelligence.
you can also pass extra/custom tags/params to be logged by returning them in a dictionary from your training function.

> note: you cannot override the parameters/metrics logged by autologging, but you can add extra tags/params.

supported autologging libraries: refer to [mlflow documentation](https://mlflow.org/docs/latest/tracking/autolog#supported-libraries)


```python
@mlflow_experiment(
    experiment_name="my_experiment",
    run_name_prefix="demo_run",
    tracking_uri="http://localhost:5000", 
    autolog_enabled=True,
    autolog_config={"log_models": True}
)
def train_my_model():
    ### your training logic here
    train_accuracy = 0.95
    test_accuracy = 0.90

    # optionally return extra tags/params for logging this works even for autolog=True
    return {
        "params": {"max_depth": 5, "n_estimators": 100},
        "metrics": {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy},
        "tags": {"model_type": "random_forest"}
    }

if __name__ == "__main__":
    train_my_model()

```

- run mlflow ui in your terminal and open http://localhost:5000 (adjust as needed).
- you will see an experiment named my_experiment with your run logs, including any parameters and metrics.


### for manual logging

```python

@mlflow_experiment("your-manual-experiment-name")
def train_custom_model():
    # manual training code
    model = ...
    model.fit(...)
    # compute metrics, etc.
    accuracy = ...
    
    #  return a dictionary with params, metrics, and tags to be logged
    return {
        "params": {"some_param": 123},
        "metrics": {"accuracy": accuracy},
        "tags": {"special_run": True},
        "artifacts": "path/to/local/artifact_dir_or_file",
        "model_local_path": "path/to/saved_model",
        "model_name": "my_registered_model"
    }

```

## feedback or contributions

this is a simple approach that works for some use cases. we would love your thoughts:

	•	is it helpful for your projects?
	•	any additional features or integrations you would like to see?
	•	found a bug or have a suggestion? open an issue or submit a pull request.

hope this decorator reduces the friction of using mlflow and helps standardize your mlops workflows. thanks for checking it out!