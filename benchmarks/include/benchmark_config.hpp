#ifndef RESHUFFLE_EXPERIMENTS_COMMON_HPP
#define RESHUFFLE_EXPERIMENTS_COMMON_HPP

namespace common {
    using SendType = double;

    constexpr auto EXPECTED_NUM_PROCESSORS = 4;

    constexpr auto START = 10;
    constexpr auto LIMIT = 100;
    constexpr auto STEP = 10;
}// namespace common

namespace gather {
    constexpr auto INITIAL_NUM_PROCESSORS_PER_DIMENSION = 2;
    constexpr auto FINAL_NUM_PROCESSORS_PER_DIMENSION = 1;
}// namespace gather

namespace scatter {
    constexpr auto INITIAL_NUM_PROCESSORS_PER_DIMENSION = 1;
    constexpr auto FINAL_NUM_PROCESSORS_PER_DIMENSION = 2;
}// namespace scatter

namespace change_block {
    constexpr auto INITIAL_NUM_PROCESSORS_PER_DIMENSION = 2;
    constexpr auto FINAL_NUM_PROCESSORS_PER_DIMENSION = 2;
    constexpr auto INITIAL_BLOCK_SIZE = 5;
    constexpr auto FINAL_BLOCK_SIZE = 10;
}// namespace change_block


static_assert(common::EXPECTED_NUM_PROCESSORS ==
              gather::INITIAL_NUM_PROCESSORS_PER_DIMENSION *
                      gather::INITIAL_NUM_PROCESSORS_PER_DIMENSION);

static_assert(common::EXPECTED_NUM_PROCESSORS ==
              scatter::FINAL_NUM_PROCESSORS_PER_DIMENSION *
                      scatter::FINAL_NUM_PROCESSORS_PER_DIMENSION);

static_assert(common::EXPECTED_NUM_PROCESSORS ==
              change_block::INITIAL_NUM_PROCESSORS_PER_DIMENSION *
                      change_block::FINAL_NUM_PROCESSORS_PER_DIMENSION);

#endif//RESHUFFLE_EXPERIMENTS_COMMON_HPP
