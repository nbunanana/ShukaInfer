/*
 * @file    activation.cpp
 * @author  nbunanana
 * @date    10/25/26
 * @brief   Interface classes for all activation functions.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "layer/details/activation.hpp"

#include "layer/details/simd.hpp"

namespace shukainfer::activations {
    std::string ActivationTypeToString(const ActivationType type) {
        std::string activate_type;
        switch (type) {
            case ActivationType::kActivationRelu: {
                activate_type = "Relu";
                break;
            }
            case ActivationType::kActivationSiLU: {
                activate_type = "SiLU";
                break;
            }
            case ActivationType::kActivationSigmoid: {
                activate_type = "Sigmoid";
                break;
            }
            case ActivationType::kActivationHardSwish: {
                activate_type = "HardSwish";
                break;
            }
            case ActivationType::kActivationHardSigmoid: {
                activate_type = "HardSigmoid";
                break;
            }
            case ActivationType::kActivationRelu6: {
                activate_type = "Relu6";
                break;
            }
            default: {
                activate_type = "Unknown";
                break;
            }
        }
        return activate_type;
    }

    template<typename T>
    StatusCode ActivationLayer<T>::Check(const std::vector<ftensorptr> &inputs, const std::vector<ftensorptr> &outputs) {
        const std::string& activation_type =ActivationTypeToString(this->act_type_);

        if (inputs.empty()) {
            LOG(ERROR) << "The inputs tensor array in the " + activation_type + " is empty.";
            return StatusCode::kInferInputsEmpty;
        }

        if (outputs.empty()) {
            LOG(ERROR) << "The outputs tensor array in the " + activation_type + " is empty.";
            return StatusCode::kInferOutputsEmpty;
        }

        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the " + activation_type + "layer do not match!!!";
            return StatusCode::kInferDimMismatch;
        }
        return StatusCode::kSuccess;
    }

    template<typename T>
    StatusCode ActivationLayer<T>::Forward(const std::vector<std::shared_ptr<Tensor<T> > > &inputs, std::vector<std::shared_ptr<Tensor<T> > > &outputs) {
        StatusCode check_status = this->Check(inputs, outputs);
        if (check_status != StatusCode::kSuccess) {
            return check_status;
        }

        const uint32_t batch_size = inputs.size();
        const std::string& act_type_str = ActivationTypeToString(this->act_type_);
        ActivationFunc activation_function = ApplySSEActivation(this->act_type_);

#pragma omp parallel for num_threads(batch_size)
        for (uint32_t i = 0;i < batch_size; i++) {
            const std::shared_ptr<Tensor<T> >& input = inputs.at(i);
            CHECK(input != nullptr && input->empty()) << "The input tensor array in the " + act_type_str + "layer has an empty tensor " << i << "th";

            std::shared_ptr<Tensor<T>>& output = outputs.at(i);
            if (output != nullptr && output->empty()) {
                output = std::make_shared<Tensor<T>>(input->shapes());
            }

            CHECK(output != nullptr && output->empty()) << "The output tensor array in the " + act_type_str + "layer has an empty tensor " << i << "th";
            activation_function(input, output);
        }
        return StatusCode::kSuccess;
    }

    template <typename T>
    ActivationLayer<T>::ActivationLayer(ActivationType type, std::string layer_name) : NonParamLayer<T>(std::move(layer_name)), act_type_(type) {}
} // shukainfer::activations
