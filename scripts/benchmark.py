import argparse

from common import plotter
from common.results import Results


def get_args():
    parser = argparse.ArgumentParser(description="Plot benchmark metrics.")

    parser.add_argument(
        "--json_file",
        type=str,
        help="Path to the JSON file with benchmark results"
    )

    parser.add_argument(
        "--metric_names",
        type=str,
        nargs='*',
        default=["cpu_time", "real_time"],
        help="List of metric names to plot, e.g., --metric_names cpu_time real_time"
    )

    parser.add_argument(
        "--group_by",
        type=str,
        default="none",
        choices=["none", "experiment", "metric"],
        help="Group the metrics by this field")

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

    json_file = args.json_file
    metric_names = args.metric_names
    group_by = args.group_by
    experiments = args.experiments

    results = Results.from_file(json_file).filter_by_metric_names(metric_names).filter_by_experiment_names(experiments)

    plotter.plot_results(results, plotter.GroupBy(group_by))


if __name__ == "__main__":
    main()
