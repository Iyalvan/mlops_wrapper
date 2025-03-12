import time
import random
from mlops_wrapper.experiments.base_experiment import BaseExperiment

class MyExperiment(BaseExperiment):
    def _execute_experiment(self, params: dict) -> dict:
        time.sleep(1)
        epochs = int(params.get("epochs", 5))
        learning_rate = float(params.get("learning_rate", 0.01))
        accuracy = random.uniform(0.7, 0.95)
        return {    
            "final_accuracy": accuracy,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "metric_name": "42"
        }