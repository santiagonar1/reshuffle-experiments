#ifndef RESHUFFLE_EXPERIMENTS_COMMON_HPP
#define RESHUFFLE_EXPERIMENTS_COMMON_HPP

namespace common {
    using SendType = double;

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

#endif//RESHUFFLE_EXPERIMENTS_COMMON_HPP
