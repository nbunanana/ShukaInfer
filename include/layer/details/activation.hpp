/*
 * @file    activation.hpp
 * @author  nbunanana
 * @date    10/25/26
 * @brief   Interface classes for all activation functions.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_LAYER_DETAILS_ACTIVATION_HPP
#define SHUKAINFER_LAYER_DETAILS_ACTIVATION_HPP
#include "data/tensor.hpp"
#include "layer/abstract/non_param_layer.hpp"

namespace shukainfer::activations {
    using ActivationFunc = std::function<void(ftensorptr, ftensorptr)>;

    enum class ActivationType {
        kActivationUnknow = -1,
        kActivationRelu = 0,
        kActivationSiLU = 1,
        kActivationSigmoid = 2,
        kActivationHardSwish = 3,
        kActivationHardSigmoid = 4,
        kActivationRelu6 = 5,
    };

    std::string ActivationTypeToString(const ActivationType type);

    template <typename T = float>
    class ActivationLayer : public NonParamLayer<T> {
    public:
        explicit ActivationLayer(ActivationType type, std::string layer_name);

        StatusCode Check(const std::vector<ftensorptr>& inputs, const std::vector<ftensorptr>& outputs) override;

        StatusCode Forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs, std::vector<std::shared_ptr<Tensor<T>>>& outputs) override;

    private:
        ActivationType act_type_ = ActivationType::kActivationUnknow;
    };
} // shukainfer::activations
#endif //SHUKAINFER_LAYER_DETAILS_ACTIVATION_HPP
