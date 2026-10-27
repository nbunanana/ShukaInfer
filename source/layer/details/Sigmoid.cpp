/*
 * @file    Sigmoid.cpp
 * @author  nbunanana
 * @date    10/27/26
 * @brief   Sigmoid layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "layer/details/Sigmoid.hpp"

#include "layer/abstract/layer_factory.hpp"

namespace shukainfer::activations {
    template<typename T>
    SigmoidLayer<T>::SigmoidLayer() : ActivationLayer<T>(ActivationType::kActivationSigmoid, "nn.Sigmoid") {}

    template<typename T>
    StatusCode SigmoidLayer<T>::Forward(const std::vector<std::shared_ptr<Tensor<T> > > &inputs, std::vector<std::shared_ptr<Tensor<T> > > &outputs) {
        return ActivationLayer<T>::Forward(inputs, outputs);
    }

    template<typename T>
    StatusCode SigmoidLayer<T>::CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T> > &op, std::shared_ptr<Layer<T> > &sigmoid_layer) {
        if (op == nullptr) {
            LOG(ERROR) << "SigmoidLayer::CreateInstance op cannot be nullptr.";
            return StatusCode::kParseNullOperator;
        }

        sigmoid_layer = std::make_shared<SigmoidLayer<T>>();
        return StatusCode::kSuccess;
    }

    template<typename T>
    LayerRegisterWarper<T> kSigmoidCreateInstance(SigmoidLayer<T>::CreateInstance, "nn.Sigmoid");
} // shukainfer::activations