import argparse

from common import plotter
from common.results import Results


def get_args():
    parser = argparse.ArgumentParser(description="Plot benchmark metrics.")

    parser.add_argument(
        "--base_file",
        type=str,
        help="Path to the JSON file with base benchmark results"
    )

    parser.add_argument(
        "--contender_files",
        nargs='+',
        type=str,
        help="Paths to the JSON files with contender benchmark results"
    )

    parser.add_argument(
        "--metric_names",
        type=str,
        nargs='*',
        default=["cpu_time", "real_time"],
        help="List of metric names to plot, e.g., --metric_names cpu_time real_time"
    )

    parser.add_argument(
        "--experiments",
        type=str,
        nargs='*',
        required=False,
        help="Show the results of the benchmarks for the specified experiments"
    )

    return parser.parse_args()


def main():
    args = get_args()

    base_file = args.base_file
    contender_files = args.contender_files
    metric_names = args.metric_names
    experiments = args.experiments

    base_results = Results.from_file(base_file).filter_by_metric_names(metric_names).filter_by_experiment_names(
        experiments)

    contender_results = []
    for contender_file in contender_files:
        contender_result = Results.from_file(contender_file).filter_by_metric_names(
            metric_names).filter_by_experiment_names(
            experiments)
        contender_results.append(contender_result)

    plotter.compare_results(base_results, contender_results)


if __name__ == "__main__":
    main()
