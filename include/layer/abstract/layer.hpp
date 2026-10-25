/*
 * @file    layer.hpp
 * @author  nbunanana
 * @date    10/7/26
 * @brief   The layer base class
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_LAYER_ABSTRACT_LAYER_HPP
#define SHUKAINFER_LAYER_ABSTRACT_LAYER_HPP

#include <glog/logging.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "data/tensor.hpp"
#include "runtime/runtime_operator.hpp"
#include "code_status.hpp"

namespace shukainfer {
    /**
     * @brief Base layer class(float)
     *
     * Abstract base class for neural network layers.
     * Defins the common layer interface and methods.
     *
     * @tparam T Data type (float, int8 etc).
     */
     template<typename T>
     class Layer{
     public:

        explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}

        virtual ~Layer() = default;

        /**
         * @brief
         * @param inputs
         * @param outputs
         * @return
         */
        virtual StatusCode Check(const std::vector<std::shared_ptr<Tensor<T>>>& inputs, const std::vector<std::shared_ptr<Tensor<T>>>& outputs);

        /**
         * @brief Performs forward inference.
         *
         * This method implement the forward inference for this layer.
         * It takes the input tensors, performs the required computation(convolution, matrix multiply etc.),
         * and produces the output tensors.
         *
         * The inputs and outputs are based on the connections in the runtime graph.
         * This method implements the core inference logic of the layer.
         *
         * @return Status code indicating success or failure(specific error with status code).
         */
        virtual StatusCode Forward();

        /**
         * @brief Performs forward inference.
         *
         * This method implement the forward inference for this layer.
         * It takes the input tensors, performs the required computation(convolution, matrix multiply etc.),
         * and produces the output tensors.
         *
         * The inputs and outputs are based on the connections in the runtime graph.
         * This method implements the core inference logic of the layer.
         *
         * @param inputs Input tensors.
         * @param outputs Output tensors.
         * @returnStatus code indicating success or failure(specific error with status code).
         */
        virtual StatusCode Forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs, std::vector<std::shared_ptr<Tensor<T>>>& outputs);

        /**
         * @brief Get layer weights.
         * @return Vector of weight tensors.
         */
        virtual const std::vector<std::shared_ptr<Tensor<T>>>& get_weights() const;

        /**
         * @brief Get layer bias.
         * @return Vector of bias tensors.
         */
        virtual const std::vector<std::shared_ptr<Tensor<T>>>& get_bias() const;

        /**
         * @brief Sets the layer weights.
         * @param weights Vector of weight tensors.
         */
        virtual void set_weights(const std::vector<std::shared_ptr<Tensor<T>>>& weights);

        /**
         * @brief Sets the layer biases.
         * @param bias Vector of bias tensors.
         */
        virtual void set_bias(const std::vector<std::shared_ptr<Tensor<T>>>& bias);

        /**
         * @brief Sets weights from float vector
         *
         * Sets the weight values from a vector of floats.
         *
         * The length of the vector should match the total number of weight values.
         *
         * @param weights Vector of weight values
         */
        virtual void set_weights(const std::vector<Tensor<T>>& weights);

        /**
        * @brief Sets biases from float vector
        *
        * Sets the bias values from a vector of floats.
        *
        * The length of the vector should match the total number of bias values.
        *
        * @param bias Vector of bias values
        */
        virtual void set_bias(const std::vector<Tensor<T>>& bias);

        /**
        * @brief Gets layer name
        *
        * @return Layer name
        */
        virtual const std::string& get_layer_name() const;

        /**
        * @brief Sets corresponding runtime operator
        *
        * Associates this layer instance with the given runtime operator.
        *
        * @param runtime_operator The RuntimeOperator for this layer
        */
        void set_runtime_operator(const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);
     protected:
        std::string layer_name_;
        std::weak_ptr<RuntimeOperatorBase<T>> runtime_operator_;
    };
    } // shukainfer
#endif //SHUKAINFER_LAYER_ABSTRACT_LAYER_HPP
