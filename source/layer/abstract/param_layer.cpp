/*
 * @file    param_layer.cpp
 * @author  nbunanana
 * @date    10/14/26
 * @brief   ${DO_NOT_EDIT}
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "layer/abstract/param_layer.hpp"

namespace shukainfer {
    template<typename T>
    ParamLayer<T>::ParamLayer(const std::string &layer_name) : Layer<T>(layer_name) {}

    template<typename T>
    void ParamLayer<T>::InitWeightsParam(uint32_t param_count, uint32_t param_channel, uint32_t param_height, uint32_t param_width) {
        this->weights_ = std::vector<std::shared_ptr<Tensor<T>>>(param_count);

        for (uint32_t i = 0; i < param_count; i++) {
            this->weights_.at(i) = std::make_shared<Tensor<T>>(param_channel, param_height, param_width);
        }
    }

    template<typename T>
    void ParamLayer<T>::InitBiasParam(uint32_t param_count, uint32_t param_channel, uint32_t param_height, uint32_t param_width) {
        this->weights_ = std::vector<std::shared_ptr<Tensor<T>>>(param_count);

        for (uint32_t i = 0; i < param_count; i++) {
            this->bias_.at(i) = std::make_shared<Tensor<T>>(param_channel, param_height, param_width);
        }
    }

    template <typename T>
    const std::vector<std::shared_ptr<Tensor<T>>>& ParamLayer<T>::get_weights() const {
        return this->weights_;
    }

    template<typename T>
    const std::vector<std::shared_ptr<Tensor<T> > > &ParamLayer<T>::get_bias() const {
        return this->bias_;
    }

    template<typename T>
    void ParamLayer<T>::set_weights(const std::vector<std::shared_ptr<Tensor<T> > > &weights) {
        CHECK(weights.size() == this->weights_.size());
        for (uint32_t i = 0; i < weights.size(); i++) {
            CHECK(this->weights_.at(i) != nullptr);
            CHECK(this->weights_.at(i)->rows() == weights.at(i)->rows());
            CHECK(this->weights_.at(i)->cols() == weights.at(i)->cols());
            CHECK(this->weights_.at(i)->channels() == weights.at(i)->channels());
        }
        this->weights_ = weights;
    }


    template<typename T>
    void ParamLayer<T>::set_bias(const std::vector<std::shared_ptr<Tensor<T> > > &bias) {
        CHECK(bias.size() == this->bias_.size());
        for (uint32_t i = 0; i < bias.size(); i++) {
            CHECK(this->bias_.at(i) != nullptr);
            CHECK(this->bias_.at(i)->rows() == bias.at(i)->rows());
            CHECK(this->bias_.at(i)->cols() == bias.at(i)->cols());
            CHECK(this->bias_.at(i)->channels() == bias.at(i)->channels());
        }
        this->bias_ = bias;
    }

    template<typename T>
    void ParamLayer<T>::set_weights(const std::vector<T> &weights) {
        size_t weight_size = 0;
        const size_t elem_size = weights.size();

        const uint32_t batch_size = weights.size();
        for (uint32_t i = 0; i < batch_size; i++) {
            weight_size += this->weights_.at(i)->size();
        }

        CHECK_EQ(weight_size, elem_size);
        CHECK_EQ(elem_size % batch_size, 0);
        const uint32_t blob_size = elem_size / batch_size;
        for (uint32_t i = 0; i < batch_size; i++) {
            const auto& sub_values = std::vector<T>(weights.begin() + i * blob_size, weights.begin() + i * blob_size + blob_size);
            this->weights_.at(i)->Fill(sub_values);
        }
    }

    template<typename T>
    void ParamLayer<T>::set_bias(const std::vector<Tensor<T> > &bias) {
        size_t bias_size = 0;
        const size_t elem_size = bias.size();

        const uint32_t batch_size = bias.size();
        for (uint32_t i = 0;i < batch_size; i++) {
            bias_size += this->bias_.at(i)->size();
        }

        CHECK_EQ(bias_size, elem_size);
        CHECK_EQ(elem_size % batch_size, 0);
        const uint32_t blob_size = elem_size / batch_size;
        for (uint32_t i =0; i < batch_size; i++) {
            const auto& sub_values = std::vector<T>(bias.begin() + i * blob_size, bias.begin() + i * blob_size + blob_size);
            this->bias_.at(i)->Fill(sub_values);
        }

    }

    template<typename T>
    std::shared_ptr<Tensor<T> > ParamLayer<T>::weight(uint32_t index) const {
        CHECK_LE(index, this->weights_.size());
        return this->weights_.at(index);
    }
} // shukainfer
