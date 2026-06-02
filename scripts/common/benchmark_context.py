class BenchmarkContext:
    def __init__(self, date, host_name, executable, num_cpus, mhz_per_cpu, cpu_scaling_enabled):
        self.date = date
        self.host_name = host_name
        self.executable = executable
        self.num_cpus = num_cpus
        self.mhz_per_cpu = mhz_per_cpu
        self.cpu_scaling_enabled = cpu_scaling_enabled
