import re


def _have_same(metrics, key_function):
    if not metrics:
        return True

    key = key_function(metrics[0])
    for metric in metrics:
        if key_function(metric) != key:
            return False

    return True


def _filter_by_experiment_name(experiment_name, metrics):
    filtered_metrics = []
    for metric in metrics:
        if re.search(experiment_name, metric.experiment_name) is not None:
            filtered_metrics.append(metric)
    return filtered_metrics


def _filter_by_metric_name(metric_name, metrics):
    filtered_metrics = []
    for metric in metrics:
        if re.search(metric_name, metric.metric_name) is not None:
            filtered_metrics.append(metric)
    return filtered_metrics


def have_same_experiment_name(metrics):
    return _have_same(metrics, lambda metric: metric.experiment_name)


def have_same_metric_name(metrics):
    return _have_same(metrics, lambda metric: metric.metric_name)


def have_same_x_units(metrics):
    return _have_same(metrics, lambda metric: metric.x_units)


def have_same_y_units(metrics):
    return _have_same(metrics, lambda metric: metric.y_units)


def filter_by_experiment_names(experiment_names, metrics):
    if not experiment_names:
        return metrics

    filtered_metrics = []
    for experiment_name in experiment_names:
        filtered_metrics.extend(_filter_by_experiment_name(experiment_name, metrics))
    return filtered_metrics


def filter_by_metric_names(metric_names, metrics):
    if not metric_names:
        return metrics

    filtered_metrics = []
    for metric_name in metric_names:
        filtered_metrics.extend(_filter_by_metric_name(metric_name, metrics))
    return filtered_metrics
