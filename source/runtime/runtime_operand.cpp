/*
 * @file    runtime_operand.cpp
 * @author  nbunanana
 * @date    9/27/26
 * @brief   Base of runtime graph operand.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#include "runtime/runtime_operand.hpp"

namespace shukainfer{
    template<typename T>
    size_t RuntimeOperandBase<T>::size() const {
            if (shapes.empty()) {
                return 0;
            }
            const size_t size = std::accumulate(shapes.begin(), shapes.end(), static_cast<size_t>(1), std::multiplies<size_t>());
            return size;
    }

    template size_t RuntimeOperandBase<float>::size() const;
    template size_t RuntimeOperandBase<int8_t>::size() const;
}// shukainfer
