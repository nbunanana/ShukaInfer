/*
 * @file    Relu6.cpp
 * @author  nbunanana
 * @date    10/28/26
 * @brief   Relu6 layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#include "layer/details/Relu6.hpp"
#include "layer_factory.hpp"

namespace shukainfer::activations {
    template<typename T>
    Relu6Layer<T>::Relu6Layer() : ActivationLayer<T>(ActivationType::kActivationRelu6, "nn.relu6"){}

    template<typename T>
    StatusCode Relu6Layer<T>::Forward(const std::vector<std::shared_ptr<Tensor<T> > > &inputs, std::vector<std::shared_ptr<Tensor<T> > > &outputs) {
        return ActivationLayer<T>::Forward(inputs, outputs);
    }

    template<typename T>
    StatusCode Relu6Layer<T>::CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T> >& op, std::shared_ptr<Layer<T> >& relu6_layer) {
        if (op == nullptr) {
            LOG(ERROR) << "The relu6 operator parameter in the layer is null pointer.";
            return StatusCode::kParseNullOperator;
        }
        relu6_layer = std::make_shared<Relu6Layer<T>>();
        return StatusCode::kSuccess;
    }

    template <typename T>
    LayerRegisterWarper<T> CreateHardSwishLayer(Relu6Layer<T>::CreateInstance, "nn.relu6");
} // shkainfer::activations
