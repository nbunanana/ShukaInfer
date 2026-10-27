/*
 * @file    HardSwith.hpp
 * @author  nbunanana
 * @date    10/27/26
 * @brief   HardSwish layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_LAYER_DETAILS_HARDSWITH_HPP
#define SHUKAINFER_LAYER_DETAILS_HARDSWITH_HPP
#include "activation.hpp"

namespace shukainfer::activations {
    template <typename T>
    class HardSwishLayer : public ActivationLayer<T> {
    public:
        explicit HardSwishLayer();

        StatusCode Forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs, std::vector<std::shared_ptr<Tensor<T>>>& outputs);

        StatusCode CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T>>& op, std::shared_ptr<Layer<T>>& hardswish_layer);
    };
}
#endif //SHUKAINFER_LAYER_DETAILS_HARDSWITH_HPP
