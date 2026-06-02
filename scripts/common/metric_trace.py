from common.metric_utils import have_same_experiment_name, have_same_metric_name, have_same_x_units, have_same_y_units


class MetricTrace:
    def __init__(self, experiment_name, metric_name, x_values, x_units, y_values, y_units):
        self.experiment_name = experiment_name
        self.metric_name = metric_name
        self.x_values = x_values
        self.x_units = x_units
        self.y_values = y_values
        self.y_units = y_units

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.__dict__)

    @classmethod
    def from_metrics(cls, metrics):
        if not metrics:
            raise ValueError("Trying to create MetricTrace from empty list of metrics.")

        if not have_same_experiment_name(metrics):
            raise ValueError("Trying to create MetricTrace from metrics with different experiment names.")

        if not have_same_metric_name(metrics):
            raise ValueError("Trying to create MetricTrace from metrics with different metric names.")

        if not have_same_x_units(metrics):
            raise ValueError("Creating MetricTrace from metrics with different x units is not supported yet.")

        if not have_same_y_units(metrics):
            raise ValueError("Creating MetricTrace from metrics with different y units is not supported yet.")

        experiment_name = metrics[0].experiment_name
        metric_name = metrics[0].metric_name
        x_units = metrics[0].x_units
        y_units = metrics[0].y_units

        x_values, y_values = [], []
        for metric in metrics:
            x_values.append(metric.x_value)
            y_values.append(metric.y_value)

        return cls(experiment_name, metric_name, x_values, x_units, y_values, y_units)
