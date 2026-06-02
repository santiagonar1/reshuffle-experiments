import unittest
from common.metric import Metric
from common.metric_trace import MetricTrace


class TestMetricTrace(unittest.TestCase):
    def test_can_create_metric_trace_from_metrics(self):
        metrics = [Metric("name", "metric", 1, "x-unit", 1, "y-unit"),
                   Metric("name", "metric", 2, "x-unit", 2, "y-unit")]
        expected_metric_trace = MetricTrace("name", "metric", [1, 2], "x-unit", [1, 2],
                                            "y-unit")
        metric_trace = MetricTrace.from_metrics(metrics)
        self.assertEqual(metric_trace, expected_metric_trace)

    def test_all_metrics_should_have_same_experiment_name(self):
        metrics_with_different_experiment_name = [Metric("name", "metric", 1, "unit", 1, "unit"),
                                                  Metric("other", "metric", 2, "unit", 2, "unit")]
        self.assertRaises(ValueError, MetricTrace.from_metrics, metrics_with_different_experiment_name)

    def test_all_metrics_should_have_same_metric_name(self):
        metrics_with_different_metric_name = [Metric("name", "metric", 1, "unit", 1, "unit"),
                                              Metric("name", "other", 2, "unit", 2, "unit")]
        self.assertRaises(ValueError, MetricTrace.from_metrics, metrics_with_different_metric_name)

    def test_all_metrics_should_have_same_x_units(self):
        metrics_with_different_x_units = [Metric("name", "metric", 1, "x-unit", 1, "unit"),
                                          Metric("name", "metric", 2, "x-other-other", 2, "unit")]
        self.assertRaises(ValueError, MetricTrace.from_metrics, metrics_with_different_x_units)

    def test_all_metrics_should_have_same_y_units(self):
        metrics_with_different_y_units = [Metric("name", "metric", 1, "unit", 1, "y-unit"),
                                          Metric("name", "metric", 2, "unit", 2, "y-other-unit")]
        self.assertRaises(ValueError, MetricTrace.from_metrics, metrics_with_different_y_units)


if __name__ == '__main__':
    unittest.main()
