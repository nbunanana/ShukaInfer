/*
 * @file    Relu.hpp
 * @author  nbunanana
 * @date    10/25/26
 * @brief   Relu layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_LAYER_DETAILS_RELU_HPP
#define SHUKAINFER_LAYER_DETAILS_RELU_HPP
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"

namespace shukainfer::activations {
template <typename T = float>
class ReluLayer : public ActivationLayer<T> {
    public:
        explicit ReluLayer();

        StatusCode Forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs, std::vector<std::shared_ptr<Tensor<T>>>& outputs) override;

        static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T>>& op, std::shared_ptr<Layer<T>>& relu_layer);
};
} // shukainfer::activations
#endif //SHUKAINFER_LAYER_DETAILS_RELU_HPP
