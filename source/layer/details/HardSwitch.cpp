/*
 * @file    HardSwitch.cpp
 * @author  nbunanana
 * @date    10/27/26
 * @brief   HardSwish layer.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "layer/abstract/layer_factory.hpp"
#include "layer/details/HardSwith.hpp"

namespace shukainfer::activations {
    template<typename T>
    HardSwishLayer<T>::HardSwishLayer() : ActivationLayer<T>(ActivationType::kActivationHardSwish, "nn.hardswish"){}

    template<typename T>
    StatusCode HardSwishLayer<T>::Forward(const std::vector<std::shared_ptr<Tensor<T> > > &inputs, std::vector<std::shared_ptr<Tensor<T> > > &outputs) {
        return ActivationLayer<T>::Forward(inputs, outputs);
    }

    template<typename T>
    StatusCode HardSwishLayer<T>::CreateInstance(const std::shared_ptr<RuntimeOperatorBase<T> > &op, std::shared_ptr<Layer<T> >& hardswish_layer) {
        if (op == nullptr) {
            LOG(ERROR) << "The hardswish operator parameter in the layer is null pointer.";
            return StatusCode::kParseNullOperator;
        }
        hardswish_layer = std::make_shared<HardSwishLayer<T>>();
        return StatusCode::kSuccess;
    }

    template <typename T>
    LayerRegisterWarper<T> CreateHardSwishLayer(HardSwishLayer<T>::CreateInstance, "nn.hardswish");
}
