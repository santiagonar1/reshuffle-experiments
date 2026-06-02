#!/bin/bash

set -e

BUILD_DIR=${1:-cmake-build-release}
NUM_PROCS=4

echo "Building in: $BUILD_DIR"

CMAKE_FLAGS="-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=conan_provider.cmake -DCMAKE_BUILD_TYPE=Release"

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "MacOS detected, adding toolchain file."
    CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_TOOLCHAIN_FILE=cmake/MacBrewLLVMToolchain.cmake"
fi

cmake -S . -B "$BUILD_DIR" $CMAKE_FLAGS
cmake --build "$BUILD_DIR" --target costa_benchmark.out scalapack_benchmark.out reshuffle_benchmark.out --parallel


BENCHMARKS=("costa_benchmark.out" "scalapack_benchmark.out" "reshuffle_benchmark.out")
RESULTS_DIR="./scripts/results/$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="$RESULTS_DIR/log.md"

mkdir -p "$RESULTS_DIR"

# Record metadata in log.md
{
  echo "# Benchmark Log"
  echo "## Metadata"
  echo "- **Compilation Time:** $(date)"
  echo "- **CMAKE_FLAGS:** $CMAKE_FLAGS"
  echo "- **Number of Processors:** $NUM_PROCS"
  echo ""
} >> "$LOG_FILE"

for BENCHMARK in "${BENCHMARKS[@]}"; do
    EXE_PATH="$BUILD_DIR/benchmarks/$BENCHMARK"
    echo "Running $BENCHMARK..."
    # Result file name based on benchmark name
    # Mapping: <name>_benchmark.out -> results_<name>.json
    NAME_PART=${BENCHMARK%_benchmark.out}
    RESULT_NAME="results_${NAME_PART}.json"
    echo "### Benchmark: $BENCHMARK" >> "$LOG_FILE"
    echo '```' >> "$LOG_FILE"
    mpirun -n $NUM_PROCS "$EXE_PATH" --benchmark_out="$RESULT_NAME" --benchmark_out_format=json 2>&1 | tee -a "$LOG_FILE"
    echo '```' >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    mv "$RESULT_NAME" "$RESULTS_DIR/"
    echo "Result saved to $RESULTS_DIR/$RESULT_NAME"
done

echo "Benchmarks completed. Results are in $RESULTS_DIR"
