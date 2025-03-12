import argparse
from examples.my_experiment import MyExperiment

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ML experiments with the standardized MLflow wrapper."
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Name of the MLflow run."
    )
    parser.add_argument(
        "--param",
        action="append",
        help="Key=Value parameters (e.g. --param epochs=10 --param learning_rate=0.01)",
        default=[]
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    params = dict(item.split("=") for item in args.param)
    experiment = MyExperiment()
    metrics = experiment.run(run_name=args.run_name, params=params)
    print("Experiment completed with metrics:", metrics)

if __name__ == "__main__":
    main()