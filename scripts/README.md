# Benchmark plotting scripts

In this folder you can find scripts to plot results produced by google benchmark. We have:

1. `benchmark.py` - plot the results of a single benchmark
2. `compare.py` - compare the results of multiple benchmarks

## Usage

Some examples are shown below. Check the help message of each scripts for more details.

```bash
python benchmark.py --json_file my_results.json --metric_names real_time --group_by metric --experiments from_one
```

```bash
python compare.py --base_file my_base_results.json --contender_files my_contender_results_1.json my_contender_results_2.json
```