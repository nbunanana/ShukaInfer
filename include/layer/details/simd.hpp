/*
 * @file    simd.hpp
 * @author  nbunanana
 * @date    10/25/26
 * @brief   
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_LAYER_DETAILS_SIMD_HPP
#define SHUKAINFER_LAYER_DETAILS_SIMD_HPP
#include "activation.hpp"

namespace shukainfer::activations {
    ActivationFunc ApplySSEActivation(ActivationType act_type);
} // shukainfer
#endif //SHUKAINFER_LAYER_DETAILS_SIMD_HPP
