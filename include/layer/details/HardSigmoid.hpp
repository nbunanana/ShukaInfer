/*
 * @file    HardSigmoid.hpp
 * @author  nbunanana
 * @date    10/27/26
 * @brief   HardSigmoid layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_HARDSIGMOID_HPP
#define SHUKAINFER_HARDSIGMOID_HPP
#include "activation.hpp"

namespace shukainfer::activations {
    template <typename T = float>
    class HardSigmoidLayer : public ActivationLayer<T> {
    public:
        explicit HardSigmoidLayer();

        StatusCode Forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs, std::vector<std::shared_ptr<Tensor<T>>>& outputs) override;

        StatusCode CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T>>& op, std::shared_ptr<Layer<T>>& sigmoid_layer);
    };
}

#endif //SHUKAINFER_HARDSIGMOID_HPP
