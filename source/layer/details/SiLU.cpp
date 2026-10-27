/*
 * @file    SiLU.cpp
 * @author  nbunanana
 * @date    10/26/26
 * @brief   SiLU layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "layer/details/SiLU.hpp"

#include "layer/abstract/layer_factory.hpp"

namespace shukainfer::activations {
    template<typename T>
    SiLULayer<T>::SiLULayer() : ActivationLayer<T>(ActivationType::kActivationSiLU, "nn.SiLU") {}

    template<typename T>
    StatusCode SiLULayer<T>::Forward(const std::vector<std::shared_ptr<Tensor<T> > > &inputs, std::vector<std::shared_ptr<Tensor<T> > > &outputs) {
        return ActivationLayer<T>::Forward(inputs, outputs);
    }

    template <typename T>
    StatusCode SiLULayer<T>::CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T>>& op, std::shared_ptr<Layer<T>>& silu_layer) {
        if (op != nullptr) {
            LOG(ERROR) << "The silu operator parameter in the layer is null pointer.";
            return StatusCode::kParseNullOperator;
        }

        silu_layer = std::make_shared<SiLULayer<T>>();
        return StatusCode::kSuccess;
    }

    template <typename T>
    LayerRegisterWarper<T> kSiLUCreateInstance(SiLULayer<T>::CreateInstance, "nn.SiLU");
} // shukainfer::activations