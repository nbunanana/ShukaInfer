/*
 * @file    SiLU.hpp
 * @author  nbunanana
 * @date    10/26/26
 * @brief   SiLU layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_LAYER_DETAILS_SILU_HPP
#define SHUKAINFER_LAYER_DETAILS_SILU_HPP
#include "activation.hpp"

namespace shukainfer::activations {
    template <typename T = float>
    class SiLULayer : public ActivationLayer<T> {
    public:
        explicit SiLULayer();

        StatusCode Forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs, std::vector<std::shared_ptr<Tensor<T>>>& outputs) override;

        static  StatusCode CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T>>& layer, std::shared_ptr<Layer<T>>& silu_layer);
    };
} // shukainfer::activations
#endif //SHUKAINFER_LAYER_DETAILS_SILU_HPP
