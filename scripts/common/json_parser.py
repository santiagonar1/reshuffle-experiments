import json
from common.benchmark_context import BenchmarkContext
from common.metric import Metric
from common.metric_trace import MetricTrace
from itertools import groupby


def _get_experiment_name(benchmark_run_name):
    return benchmark_run_name.split("/")[0]


def _get_num_elements(benchmark_run_name):
    return int(benchmark_run_name.split("/")[1])


def _parse_time_metric(benchmark, time_metric_name):
    experiment_name = _get_experiment_name(benchmark["run_name"])
    num_elements = _get_num_elements(benchmark["run_name"])
    time_unit = benchmark["time_unit"]
    value = float(benchmark[time_metric_name])
    return Metric(experiment_name, time_metric_name, num_elements, "num_elements", value, time_unit)


def _group_by(metrics, key_function):
    sorted_metrics = sorted(metrics, key=key_function)
    grouped = {}

    for key, metrics in groupby(sorted_metrics, key=key_function):
        grouped[key] = list(metrics)

    return grouped


def group_by_experiment(metrics):
    return _group_by(metrics, key_function=lambda metric: metric.experiment_name)


def group_by_metric_name(metrics):
    return _group_by(metrics, key_function=lambda metric: metric.metric_name)


class JSONParser:
    def __init__(self, json_data):
        self.json_data = json.loads(json_data)

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, "r") as f:
            return cls(f.read())

    def parse_benchmark_context(self):
        return BenchmarkContext(
            self.json_data["context"]["date"],
            self.json_data["context"]["host_name"],
            self.json_data["context"]["executable"],
            self.json_data["context"]["num_cpus"],
            self.json_data["context"]["mhz_per_cpu"],
            self.json_data["context"]["cpu_scaling_enabled"]
        )

    def _parse_time_metrics(self, time_metric_names):
        benchmarks = self.json_data["benchmarks"]
        time_metrics = []
        for benchmark in benchmarks:
            for metric_name in time_metric_names:
                time_metrics.append(_parse_time_metric(benchmark, metric_name))
        return time_metrics

    def parse_time_metric_traces(self, time_metric_names):
        time_metrics = self._parse_time_metrics(time_metric_names)

        metric_traces = []
        for metrics_grouped_by_experiment in group_by_experiment(time_metrics).values():
            for metrics in group_by_metric_name(metrics_grouped_by_experiment).values():
                metric_traces.append(MetricTrace.from_metrics(metrics))

        return metric_traces

    def parse(self):
        context = self.parse_benchmark_context()

        return context
