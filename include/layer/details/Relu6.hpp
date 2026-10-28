/*
 * @file    Relu6.hpp
 * @author  nbunanana
 * @date    10/28/26
 * @brief   Relu6 layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_LAYER_DETAILS_RELU6_HPP
#define SHUKAINFER_LAYER_DETAILS_RELU6_HPP
#include "activation.hpp"

namespace shukainfer::activations {
    template <typename T = float>
    class Relu6Layer : ActivationLayer<T> {
    public:
        explicit Relu6Layer();

        StatusCode Forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs, std::vector<std::shared_ptr<Tensor<T>>>& outputs) override;

        StatusCode CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T>>& op,  std::shared_ptr<Layer<T>>& relu6_layer);
    };
} // shukainfer::activations
#endif //SHUKAINFER_LAYER_DETAILS_RELU6_HPP
