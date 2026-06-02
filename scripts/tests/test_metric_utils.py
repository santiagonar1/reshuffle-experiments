import unittest

from common.metric import Metric
from common.metric_utils import have_same_experiment_name, have_same_metric_name, have_same_x_units, have_same_y_units, \
    filter_by_experiment_names, filter_by_metric_names


class TestMetricUtils(unittest.TestCase):
    def test_have_same_experiment_name_returns_true_if_all_metrics_have_same_experiment_name(self):
        same_experiment_name_metrics = [Metric("experiment", "metric", 1, "unit", 1, "unit"),
                                        Metric("experiment", "metric", 2, "unit", 2, "unit")]

        self.assertTrue(have_same_experiment_name(same_experiment_name_metrics))

    def test_have_same_experiment_name_returns_false_if_all_metrics_not_have_same_experiment_name(self):
        different_experiment_name_metrics = [Metric("experiment", "metric", 1, "unit", 1, "unit"),
                                             Metric("other", "metric", 2, "unit", 2, "unit")]

        self.assertFalse(have_same_experiment_name(different_experiment_name_metrics))

    def test_have_same_metric_name_returns_true_if_all_metrics_have_same_metric_name(self):
        same_metric_name_metrics = [Metric("name", "metric", 1, "unit", 1, "unit"),
                                    Metric("name", "metric", 2, "unit", 2, "unit")]

        self.assertTrue(have_same_metric_name(same_metric_name_metrics))

    def test_have_same_metric_name_returns_false_if_not_all_metrics_have_same_metric_name(self):
        different_metric_name_metrics = [Metric("name", "metric", 1, "unit", 1, "unit"),
                                         Metric("name", "other", 2, "unit", 2, "unit")]

        self.assertFalse(have_same_metric_name(different_metric_name_metrics))

    def test_have_same_x_units_returns_true_if_all_metrics_have_same_x_units(self):
        same_x_units_metrics = [Metric("name", "metric", 1, "x-unit", 1, "unit"),
                                Metric("name", "metric", 2, "x-unit", 2, "unit")]

        self.assertTrue(have_same_x_units(same_x_units_metrics))

    def test_have_same_x_units_returns_false_if_not_all_metrics_have_same_x_units(self):
        different_x_units_metrics = [Metric("name", "metric", 1, "x-unit", 1, "unit"),
                                     Metric("name", "metric", 2, "x-other-unit", 2, "unit")]

        self.assertFalse(have_same_x_units(different_x_units_metrics))

    def test_have_same_y_units_returns_true_if_all_metrics_have_same_y_units(self):
        same_y_units_metrics = [Metric("name", "metric", 1, "unit", 1, "y-unit"),
                                Metric("name", "metric", 2, "unit", 2, "y-unit")]

        self.assertTrue(have_same_y_units(same_y_units_metrics))

    def test_have_same_y_units_returns_false_if_not_all_metrics_have_same_y_units(self):
        different_y_units_metrics = [Metric("name", "metric", 1, "unit", 1, "y-unit"),
                                     Metric("name", "metric", 2, "unit", 2, "y-other-unit")]

        self.assertFalse(have_same_y_units(different_y_units_metrics))

    def test_filter_metrics_by_experiment_names_returns_metrics_if_experiment_name_matches_any_of_the_expressions(self):
        metrics = [Metric("this-should-match", "metric", 1, "unit", 1, "unit"),
                   Metric("another-match", "metric", 2, "unit", 2, "unit"),
                   Metric("not-matching", "metric", 3, "unit", 3, "unit")]
        expected = [Metric("this-should-match", "metric", 1, "unit", 1, "unit"),
                    Metric("another-match", "metric", 2, "unit", 2, "unit")]

        self.assertEqual(filter_by_experiment_names(["this-should", "another"], metrics), expected)

    def test_filter_metrics_by_metric_names_returns_metrics_if_metric_name_matches_any_of_the_expressions(self):
        metrics = [Metric("name", "this-should-match", 1, "unit", 1, "unit"),
                   Metric("name", "another-match", 2, "unit", 2, "unit"),
                   Metric("name", "not-matching", 3, "unit", 3, "unit")]
        expected = [Metric("name", "this-should-match", 1, "unit", 1, "unit"),
                    Metric("name", "another-match", 2, "unit", 2, "unit")]

        self.assertEqual(filter_by_metric_names(["this-should", "another"], metrics), expected)


if __name__ == '__main__':
    unittest.main()
