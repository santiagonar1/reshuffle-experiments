class Metric:
    def __init__(self, experiment_name, metric_name, x_value, x_units, y_value, y_units):
        self.experiment_name = experiment_name
        self.metric_name = metric_name
        self.x_value = x_value
        self.x_units = x_units
        self.y_value = y_value
        self.y_units = y_units

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.__dict__)
