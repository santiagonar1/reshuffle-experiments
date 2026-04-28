#ifndef NULL_REPORTER_HPP
#define NULL_REPORTER_HPP

#include <benchmark/benchmark.h>


// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter final : public benchmark::BenchmarkReporter {
public:
    NullReporter() = default;
    bool ReportContext(const Context &) override { return true; }
    void ReportRuns(const std::vector<Run> &) override {}
    void Finalize() override {}
};


#endif//NULL_REPORTER_HPP
