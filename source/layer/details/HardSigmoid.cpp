/*
 * @file    HardSigmoid.cpp
 * @author  nbunanana
 * @date    10/27/26
 * @brief   HardSigmoid layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#include "layer/details/HardSigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace shukainfer::activations {
    template<typename T>
    HardSigmoidLayer<T>::HardSigmoidLayer() : ActivationLayer<T>(ActivationType::kActivationHardSwish, "nn.hardsigmoid"){}

    template<typename T>
    StatusCode HardSigmoidLayer<T>::Forward(const std::vector<std::shared_ptr<Tensor<T> > > &inputs, std::vector<std::shared_ptr<Tensor<T> > > &outputs) {
        return ActivationLayer<T>::Forward(inputs, outputs);
    }

    template<typename T>
    StatusCode HardSigmoidLayer<T>::CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T> > &op, std::shared_ptr<Layer<T> >& hardsigmoid_layer) {
        if (op == nullptr) {
            LOG(ERROR) << "The hardsigmoid operator parameter in the layer is null pointer.";
            return StatusCode::kParseNullOperator;
        }
        hardsigmoid_layer = std::make_shared<HardSigmoidLayer<T>>();
        return StatusCode::kSuccess;
    }

    template <typename T>
    LayerRegisterWarper<T> CreateHardSwishLayer(HardSigmoidLayer<T>::CreateInstance, "nn.hardsigmoid");
} // shukainfer::activations