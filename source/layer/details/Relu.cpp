/*
 * @file    Relu.cpp
 * @author  nbunanana
 * @date    10/25/26
 * @brief   Relu layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "layer/details/Relu.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace shukainfer::activations {
    template<typename T>
    ReluLayer<T>::ReluLayer() : ActivationLayer<T>(ActivationType::kActivationRelu, "nn.Relu") {}

    template<typename T>
    StatusCode ReluLayer<T>::Forward(const std::vector<std::shared_ptr<Tensor<T> > > &inputs, std::vector<std::shared_ptr<Tensor<T> > > &outputs) {
        return ActivationLayer<T>::Forward(inputs, outputs);
    }

    template<typename T>
    StatusCode ReluLayer<T>::CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T>> &op, std::shared_ptr<Layer<T> > &relu_layer) {
        if (op == nullptr) {
            LOG(ERROR) << "The relu operator parameter in the layer is null pointer!!";
            return StatusCode::kParseNullOperator;
        }

        relu_layer = std::make_shared<ReluLayer<T>>();
        return StatusCode::kSuccess;
    }

    // This is a global object being created, not a function!!!
    template <typename T>
    LayerRegisterWarper<T> kReluCreateInstance(ReluLayer<T>::CreateInstance, "nn.Relu");
} // shukainfer::activations