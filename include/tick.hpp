/*
 * @file    tick.hpp
 * @author  nbunanana
 * @date    10/24/26
 * @brief   
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_TICK_HPP
#define SHUKAINFER_TICK_HPP
#include <chrono>
#include <iostream>

#ifndef __ycm__
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x)                                                     \
printf("%s: %lfs\n", #x,                                          \
std::chrono::duration_cast<std::chrono::duration<double>>( \
std::chrono::steady_clock::now() - bench_##x)          \
.count());
#else
#define TICK(x)
#define TOCK(x)
#endif

#endif //SHUKAINFER_TICK_HPP
