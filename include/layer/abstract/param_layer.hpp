/*
 * @file    param_layer.hpp
 * @author  nbunanana
 * @date    10/13/26
 * @brief   
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#include "layer/abstract/layer.hpp"

#ifndef SHUKAINFER_LAYER_ABSTRACT_PARAMLAYER_HPP
#define SHUKAINFER_LAYER_ABSTRACT_PARAMLAYER_HPP
namespace shukainfer {
    template <typename T>
    class ParamLayer : public Layer<T> {
     public:
        explicit ParamLayer(const std::string& layer_name);

        /**
         * @brief Initializes weight tensors
         *
         * Initializes the shapes of the weight tensors based on provided dimensions.
         * Does not initialize values.
         *
         * @param param_count Number of kernels.
         * @param param_channel Channels of kernels.
         * @param param_height Height of kernels.
         * @param param_width width of kernels.
         */
        void InitWeightsParam(uint32_t param_count, uint32_t param_channel, uint32_t param_height, uint32_t param_width);

        /**
         * @brief Initializes bia tensors
         *
         * Initializes the shapes of the bia tensors based on provided dimensions.
         * Does not initialize values.
         *
         * @param param_count Number of kernels.
         * @param param_channel Channels of kernels.
         * @param param_height Height of kernels.
         * @param param_width width of kernels.
         */
        void InitBiasParam(uint32_t param_count, uint32_t param_channel, uint32_t param_height, uint32_t param_width);

        /**
         * @brief Gets the layer weight tensors.
         *
         * Override the bae class Layer<T> get_weights() method.
         * Return a reference to the weight tensors for this layer.
         *
         * @return Constant reference to the weight tensors.
         */
        const std::vector<std::shared_ptr<Tensor<T>>>& get_weights() const override;

        /**
         * @brief Gets the layer bia tensors.
         *
         * Override the bae class Layer<T> get_bias() method.
         * Return a reference to the bia tensors for this layer.
         *
         * @return Constant reference to the bia tensors.
         */
        const std::vector<std::shared_ptr<Tensor<T>>>& get_bias() const override;

        /**
         * @brief Sets the weights values.
         *
         * Override the base class Layer<T> set_weights() method.
         * Sets the weights tensor values from a vector of floats.
         * The vector length should match the total number of values.
         *
         * @param weights Vector of weights values.
         */
        void set_weights(const std::vector<T>& weights) override;

        /**
         * @brief Sets the bias values.
         *
         * Override the base class Layer<T> set_bias() method.
         * Sets the bias tensor values from a vector of floats.
         * The vector length should match the total number of values.
         *
         * @param bias Vector of bias values.
         */
        void set_bias(const std::vector<Tensor<T>>& bias) override;

        /**
          * @brief Sets the weight tensors
          *
          * Overrides the base class method.
          *
          * @param weights Weight tensors to set
          */
        void set_weights(const std::vector<std::shared_ptr<Tensor<T>>>& weights) override;

        /**
         * @brief Sets the bias tensors
         *
         * Overrides the base class method.
         *
         * @param bias Bias tensors to set
         */
        void set_bias(const std::vector<std::shared_ptr<Tensor<T>>>& bias) override;

        /**
         * @brief Gets Value of offset.
         * @param index The offset relative to the first address of weight.
         * @return Value of offset.
         */
        std::shared_ptr<Tensor<T>> weight(uint32_t index) const;

     protected:
        std::vector<std::shared_ptr<Tensor<T>>> weights_;
        std::vector<std::shared_ptr<Tensor<T>>> bias_;
    };
} // shukainfer

#endif //SHUKAINFER_LAYER_ABSTRACT_PARAMLAYER_HPP
