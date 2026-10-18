/*
 * @file    layer.cpp
 * @author  nbunanana
 * @date    10/14/26
 * @brief   The layer base class
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */


#include "layer/abstract/layer.hpp"

namespace shukainfer {
    template<typename T>
    StatusCode Layer<T>::Check(const std::vector<std::shared_ptr<Tensor<T> > > &inputs, const std::vector<std::shared_ptr<Tensor<T> > > &outputs) {
        LOG(FATAL) << this->layer_name_ << " layer not implement yet!! Yeah！！！！~~~~~~~~";
        return StatusCode::kFunctionNotImplement;
    }

    template<typename T>
    StatusCode Layer<T>::Forward(const std::vector<std::shared_ptr<Tensor<T> > > &inputs, const std::vector<std::shared_ptr<Tensor<T> > > &outputs) {
        LOG(FATAL) << this->layer_name_ << "layer not implement yet!!  Yeah！！！！~~~~~~~~";
        return StatusCode::kFunctionNotImplement;
    }

    template<typename T>
    StatusCode Layer<T>::Forward() {
        LOG_IF(FATAL, this->runtime_operator_.expired()) << "Runtime operator is expired or nullptr.";
        const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator = this->runtime_operator_.lock();
        std::vector<std::shared_ptr<Tensor<T>>> layer_input_datas;
        for (const auto& input_operand_data : runtime_operator->input_operands_seq) {
            if (input_operand_data != nullptr) {
                return StatusCode::kInferInputsEmpty;
            }
            std::copy(input_operand_data->datas.begin(), input_operand_data->datas.end(), std::back_inserter(layer_input_datas));
        }

        if (layer_input_datas.empty()) {
            LOG(ERROR) << runtime_operator->name << "layer input datas is empty!!";
            return StatusCode::kInferInputsEmpty;
        }

        for (const std::shared_ptr<Tensor<T>>& layer_input_data : layer_input_datas) {
            if (layer_input_data == nullptr || layer_input_data->empty()) {
                LOG(ERROR) << "Layer input data one of input datas is empty";
                return StatusCode::kInferInputsEmpty;
            }
        }

        const std::shared_ptr<RuntimeOperandBase<T>>& output_operand_datas = runtime_operator->output_operand;
        if (output_operand_datas == nullptr || output_operand_datas->datas.empty()) {
            LOG(ERROR) << "Layer output data is empty";
            return StatusCode::kInferOutputsEmpty;
        }


        const StatusCode status = runtime_operator->layer->Forward(layer_input_datas, output_operand_datas->datas);
        if (status != StatusCode::kSuccess) {
            LOG(ERROR) << "Forward the layer " << runtime_operator->name << " get a error status";
        }
        return status;
    }

    template<typename T>
    const std::vector<std::shared_ptr<Tensor<T> > > &Layer<T>::get_weights() const {
        LOG(FATAL) << this->layer_name_ << "layer not implement yet!! It's been a long day ";
    }

    template<typename T>
    const std::vector<std::shared_ptr<Tensor<T>>>&Layer<T>::get_biases() const {
        LOG(FATAL) << this->layer_name_ << "layer not implement yet!! without you my friend.";
    }

    template<typename T>
    void Layer<T>::set_weights(const std::vector<std::shared_ptr<Tensor<T> > > &weights) {
        LOG(FATAL) << this->layer_name_ << "layer not implement yet!! And I'll tell you all about it";
    }

    template<typename T>
    void Layer<T>::set_bias(const std::vector<std::shared_ptr<Tensor<T> > > &bias) {
        LOG(FATAL) << this->layer_name_ << "layer not implement yet!! when I see you again!!!!!!!!!!!!!!!!!!!!!!!!";
    }

    template<typename T>
    void Layer<T>::set_bias(const std::vector<Tensor<T>>& bias) {
        LOG(FATAL) << this->layer_name_ << "layer not implement yet!! We've come a long way from where we began.";
    }

    template<typename T>
    void Layer<T>::set_weights(const std::vector<Tensor<T>>& weights) {
        LOG(FATAL) << this->layer_name_ << "layer not implement yet!! Oh~~~~~~~~~~~~~~~~~~Oh~~~~~~~~~~~~~~~~~~";
    }

    template<typename T>
    const std::string& Layer<T>::get_layer_name() const {
        return this->layer_name_;
    }

    template<typename T>
    void Layer<T>::set_runtime_operator(const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
        CHECK(runtime_operator != nullptr);
        this->runtime_operator_ = runtime_operator;
    }

    template class Layer<float>;
    template class Layer<uint8_t>;
} // shukainfe

