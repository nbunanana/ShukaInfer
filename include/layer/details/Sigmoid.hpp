/*
 * @file    Sigmoid.hpp
 * @author  nbunanana
 * @date    10/27/26
 * @brief   Sigmoid layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_LAYER_DETAILS_SIGMOID_HPP
#define SHUKAINFER_LAYER_DETAILS_SIGMOID_HPP
#include "activation.hpp"
#include "layer/abstract/non_param_layer.hpp"

namespace shukainfer::activations {
    template <typename T = float>
    class SigmoidLayer : public ActivationLayer<T> {
    public:
        explicit SigmoidLayer();

        StatusCode Forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs, std::vector<std::shared_ptr<Tensor<T>>>& outputs) override;

        StatusCode CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T>>& op, std::shared_ptr<Layer<T>>& sigmoid_layer);
    };
} // shukainfer::activations
#endif //SHUKAINFER_LAYER_DETAILS_SIGMOID_HPP
