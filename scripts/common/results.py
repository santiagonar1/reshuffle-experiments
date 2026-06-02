import os

from common.json_parser import JSONParser
from common.metric_utils import filter_by_metric_names, filter_by_experiment_names


def _beautify_file_name(file_name):
    pretty_file_name = os.path.basename(file_name).replace(".json", "")
    return pretty_file_name


class Results:
    def __init__(self, file_name, metric_traces):
        self.file_name = file_name
        self.metric_traces = metric_traces

    @classmethod
    def from_file(cls, file_name):
        all_time_metrics = ["cpu_time", "real_time"]
        metric_traces = JSONParser.from_file(file_name).parse_time_metric_traces(all_time_metrics)

        return cls(_beautify_file_name(file_name), metric_traces)

    def filter_by_metric_names(self, metric_names):
        metric_traces = filter_by_metric_names(metric_names, self.metric_traces)
        file_name = self.file_name
        return Results(file_name, metric_traces)

    def filter_by_experiment_names(self, experiment_names):
        metric_traces = filter_by_experiment_names(experiment_names, self.metric_traces)
        file_name = self.file_name
        return Results(file_name, metric_traces)
