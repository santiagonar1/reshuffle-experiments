import matplotlib.pyplot as plt
from common.json_parser import group_by_experiment, group_by_metric_name
from common.metric_utils import have_same_x_units, have_same_y_units, have_same_experiment_name, have_same_metric_name
from enum import Enum


class GroupBy(Enum):
    EXPERIMENT = "experiment"
    METRIC = "metric"
    NONE = "none"


def _plot_metric_trace(metric_trace):
    plt.plot(metric_trace.x_values, metric_trace.y_values)

    plt.title(f"{metric_trace.experiment_name}")
    plt.xlabel(f"{metric_trace.x_units}")
    plt.ylabel(f"{metric_trace.metric_name} ({metric_trace.y_units})")
    plt.show()


def _plot_metric_traces_by_experiment(metric_traces):
    for experiment_name, metric_traces_same_experiment in group_by_experiment(metric_traces).items():
        for metric_trace in metric_traces_same_experiment:
            plt.plot(metric_trace.x_values, metric_trace.y_values, label=f"{metric_trace.metric_name}")

        plt.title(f"{experiment_name}")
        plt.xlabel(f"{metric_traces[0].x_units}")
        plt.ylabel(f"time ({metric_traces[0].y_units})")
        plt.legend()
        plt.show()


def _plot_metric_traces_by_metric_name(metric_traces):
    for metric_name, metric_traces_same_metric_name in group_by_metric_name(metric_traces).items():
        for metric_trace in metric_traces_same_metric_name:
            plt.plot(metric_trace.x_values, metric_trace.y_values, label=f"{metric_trace.experiment_name}")

        plt.title(f"{metric_name}")
        plt.xlabel(f"{metric_traces[0].x_units}")
        plt.ylabel(f"{metric_name} ({metric_traces[0].y_units})")
        plt.legend()
        plt.show()


def _plot_metric_traces(metric_traces):
    for metric_trace in metric_traces:
        _plot_metric_trace(metric_trace)


def plot_metric_traces(metric_traces, group_by):
    if not have_same_x_units(metric_traces):
        raise ValueError("Plotting metric traces with different x units is not supported yet.")

    if not have_same_y_units(metric_traces):
        raise ValueError("Plotting metric traces with different y units is not supported yet.")

    if group_by is GroupBy.EXPERIMENT:
        _plot_metric_traces_by_experiment(metric_traces)
    elif group_by is GroupBy.METRIC:
        _plot_metric_traces_by_metric_name(metric_traces)
    elif group_by is GroupBy.NONE:
        _plot_metric_traces(metric_traces)


def plot_results(results, group_by):
    plot_metric_traces(results.metric_traces, group_by)


def _compare_metric_traces(base_metric_trace, base_file_name, contender_metric_traces, contender_file_names):
    plt.plot(base_metric_trace.x_values, base_metric_trace.y_values, label=f"base: {base_file_name}")
    for num_contender, contender_metric_trace in enumerate(contender_metric_traces):
        plt.plot(contender_metric_trace.x_values, contender_metric_trace.y_values,
                 label=f"{contender_file_names[num_contender]}")

    plt.title(f"{base_metric_trace.experiment_name}")
    plt.xlabel(f"{base_metric_trace.x_units}")
    plt.ylabel(f"{base_metric_trace.metric_name} ({base_metric_trace.y_units})")
    plt.legend()
    plt.show()


def _append_dictionaries(dict1, dict2):
    appended_dict = {}
    for key, value in dict2.items():
        appended_dict[key] = dict1.get(key, []) + value

    return appended_dict


def compare_results(base_results, contender_results):
    base_metric_traces_by_experiment_name = group_by_experiment(base_results.metric_traces)
    contender_metric_traces_by_experiment_name = {}
    for contender_result in contender_results:
        contender_metric_traces_by_experiment_name = _append_dictionaries(contender_metric_traces_by_experiment_name,
                                                                          group_by_experiment(
                                                                              contender_result.metric_traces))

    base_file_name = base_results.file_name
    contender_file_names = [contender_result.file_name for contender_result in contender_results]

    for experiment_name, base_metric_traces in base_metric_traces_by_experiment_name.items():
        contender_metric_traces_by_metric_name = group_by_metric_name(
            contender_metric_traces_by_experiment_name[experiment_name])
        for base_metric_trace in base_metric_traces:
            metric_name = base_metric_trace.metric_name
            _compare_metric_traces(base_metric_trace, base_file_name,
                                   contender_metric_traces_by_metric_name[metric_name], contender_file_names)
