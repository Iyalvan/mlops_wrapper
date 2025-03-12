from mlops_wrapper.mlflow_wrapper import MLflowWrapper

class BaseExperiment:
    def __init__(self, experiment_name="default-experiment"):
        self.mlflow_wrapper = MLflowWrapper(experiment_name)

    def run(self, run_name, params: dict):
        """
        base method to run an experiment:
        1. start an mlflow run.
        2. log parameters.
        3. execute the experiment logic.
        4. log metrics/artifacts as needed.
        """
        with self.mlflow_wrapper.start_run(run_name=run_name):
            for key, value in params.items():
                self.mlflow_wrapper.log_params({key: value})
            metrics = self._execute_experiment(params)
            for k, v in metrics.items():
                self.mlflow_wrapper.log_metrics({k: v})
            return metrics

    def _execute_experiment(self, params: dict) -> dict:
        """
        to be implemented by child classes.
        should return a dictionary of metric names and values.
        """
        raise NotImplementedError("please implement the _execute_experiment method in your experiment class.") 