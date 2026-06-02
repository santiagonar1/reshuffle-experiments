import textwrap
import unittest
from common import json_parser
from common.json_parser import JSONParser
from common.metric import Metric
from common.metric_trace import MetricTrace


class TestJSONParser(unittest.TestCase):
    def setUp(self):
        self.json_data = textwrap.dedent("""\
        {
          "context": {
            "date": "2025-08-09T09:51:38+02:00",
            "host_name": "Mac.fritz.box",
            "executable": "reshuffle_benchmark.out",
            "num_cpus": 14,
            "mhz_per_cpu": 24,
            "cpu_scaling_enabled": false,
            "caches": [
              {
                "type": "Data",
                "level": 1,
                "size": 65536,
                "num_sharing": 0
              },
              {
                "type": "Instruction",
                "level": 1,
                "size": 131072,
                "num_sharing": 0
              },
              {
                "type": "Unified",
                "level": 2,
                "size": 4194304,
                "num_sharing": 1
              }
            ],
            "load_avg": [
              2.62402,
              2.66602,
              2.84473
            ],
            "library_version": "v1.9.1",
            "library_build_type": "release",
            "json_schema_version": 1
          },
          "benchmarks": [
            {
              "name": "shuffle_from_N_to_one/1000/manual_time",
              "family_index": 0,
              "per_family_instance_index": 0,
              "run_name": "shuffle_from_N_to_one/1000/manual_time",
              "run_type": "iteration",
              "repetitions": 1,
              "repetition_index": 0,
              "threads": 1,
              "iterations": 70032,
              "real_time": 1.1,
              "cpu_time": 1.2,
              "time_unit": "ns"
            },
            {
              "name": "shuffle_from_N_to_one/51000/manual_time",
              "family_index": 0,
              "per_family_instance_index": 1,
              "run_name": "shuffle_from_N_to_one/51000/manual_time",
              "run_type": "iteration",
              "repetitions": 1,
              "repetition_index": 0,
              "threads": 1,
              "iterations": 12447,
              "real_time":2.1,
              "cpu_time": 2.2,
              "time_unit": "ns"
            },
            {
              "name": "shuffle_from_one_to_N/501000/manual_time",
              "family_index": 0,
              "per_family_instance_index": 10,
              "run_name": "shuffle_from_one_to_N/501000/manual_time",
              "run_type": "iteration",
              "repetitions": 1,
              "repetition_index": 0,
              "threads": 1,
              "iterations": 946,
              "real_time": 3.1,
              "cpu_time": 3.2,
              "time_unit": "ns"
            }
          ]
        }
        """)

    def test_json_parser_returns_benchmark_context(self):
        parser = JSONParser(self.json_data)
        benchmark_context = parser.parse_benchmark_context()
        self.assertEqual(benchmark_context.date, "2025-08-09T09:51:38+02:00")
        self.assertEqual(benchmark_context.host_name, "Mac.fritz.box")
        self.assertEqual(benchmark_context.executable, "reshuffle_benchmark.out")
        self.assertEqual(benchmark_context.num_cpus, 14)
        self.assertEqual(benchmark_context.mhz_per_cpu, 24)
        self.assertEqual(benchmark_context.cpu_scaling_enabled, False)

    def test_json_parser_returns_time_metrics(self):
        parser = JSONParser(self.json_data)
        time_metrics = parser._parse_time_metrics(["real_time", "cpu_time"])
        expected_time_metrics = [
            Metric("shuffle_from_N_to_one", "real_time", 1000, "num_elements", 1.1, "ns"),
            Metric("shuffle_from_N_to_one", "cpu_time", 1000, "num_elements", 1.2, "ns"),
            Metric("shuffle_from_N_to_one", "real_time", 51000, "num_elements", 2.1, "ns"),
            Metric("shuffle_from_N_to_one", "cpu_time", 51000, "num_elements", 2.2, "ns"),
            Metric("shuffle_from_one_to_N", "real_time", 501000, "num_elements", 3.1, "ns"),
            Metric("shuffle_from_one_to_N", "cpu_time", 501000, "num_elements", 3.2, "ns")]
        self.assertEqual(time_metrics, expected_time_metrics)

    def test_json_parser_returns_time_metric_traces(self):
        parser = JSONParser(self.json_data)
        time_metric_traces = parser.parse_time_metric_traces(["real_time", "cpu_time"])
        expected_time_metric_traces = [
            MetricTrace("shuffle_from_N_to_one", "cpu_time", [1000, 51000], "num_elements",
                        [1.2, 2.2], "ns"),
            MetricTrace("shuffle_from_N_to_one", "real_time", [1000, 51000], "num_elements",
                        [1.1, 2.1], "ns"),
            MetricTrace("shuffle_from_one_to_N", "cpu_time", [501000], "num_elements", [3.2],
                        "ns"),
            MetricTrace("shuffle_from_one_to_N", "real_time", [501000], "num_elements", [3.1],
                        "ns")]

        self.assertEqual(time_metric_traces, expected_time_metric_traces)

    def test_get_experiment_name_works(self):
        benchmark_run_name = "shuffle_from_N_to_one/1000/manual_time"
        experiment_name = json_parser._get_experiment_name(benchmark_run_name)
        self.assertEqual(experiment_name, "shuffle_from_N_to_one")

    def test_get_num_elements_works(self):
        benchmark_run_name = "shuffle_from_N_to_one/1000/manual_time"
        num_elements = json_parser._get_num_elements(benchmark_run_name)
        self.assertEqual(num_elements, 1000)


if __name__ == '__main__':
    unittest.main()
