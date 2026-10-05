/*
 * @file   tensor_util.hpp
 * @author Dyc
 * @date   2026-09-23
 * @brief
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#ifndef SHUKAINFER_DATA_TENSORUTIL_HPP
#define SHUKAINFER_DATA_TENSORUTIL_HPP
#include "data/tensor.hpp"
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace shukainfer {

/**
 * @brief Creates a 3D tensor
 * 
 * @tparam T 
 * @param channels Number of channels
 * @param rows Number of rows
 * @param cols Number of cols
 * @return New 3D tensor
 */
template<typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols);

/**
 * @brief Creates a 2D tensor
 * 
 * @tparam T 
 * @param rows Number of rows
 * @param cols Number of cols
 * @return New 2D tensor
 */
template<typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t rows, uint32_t cols);

/**
 * @brief Creates a 1D tensor
 * 
 * @tparam T 
 * @param cols Number of cols
 * @return New 1D tensor
 */
template<typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t size);

/**
 * @brief Creates tensor with shape
 * 
 * @tparam T 
 * @param shapes Tensor dimensions
 * @return New tensor
 */
template<typename T>
std::shared_ptr<Tensor<T>> TensorCreate(const std::vector<uint32_t>& shapes);


/**
 * @brief Element-wise tensor add
 * 
 * @tparam T 
 * @param tensor1 tensor1 
 * @param tensor2 tensor2 
 * @param threshold Tolerance
 * @return true if equal within tolerance
 * @return false if not equal within tolerance
 */
template<typename T>
bool TensorIsSame(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2, T threshold = 1e-5f);

/**
 * @brief Pads a tensor
 * 
 * @tparam T 
 * @param tensor Tensor to pad
 * @param pads Padding amount for dims
 * @param padding_value Padding value
 * @return Padded tensor  
 */
template<typename T>
std::shared_ptr<Tensor<T>> TensorPadding(const std::shared_ptr<Tensor<T>> tensor, const std::vector<uint32_t>& pads, T padding_value);

/**
 * @brief Element-wise tensor add 
 * 
 * @tparam T 
 * @param tensor1 tensor1 
 * @param tensor2 tensor2 
 * @return  Output tensor 
 */
template<typename T>
std::shared_ptr<Tensor<T>> TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2);

/**
 * @brief In-place element-wise tensor add
 * 
 * @tparam T 
 * @param tensor1 tensor1 
 * @param tensor2 tensor2  
 * @param output_tensor Output
 */
template<typename T>
void TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2, const std::shared_ptr<Tensor<T>>& output_tensor);

/**
 * @brief Element-wise tensor multiply 
 * 
 * @tparam T 
 * @param tensor1 tensor1 
 * @param tensor2 tensor2 
 * @return Output
 */
template<typename T>
std::shared_ptr<Tensor<T>> TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2);

/**
 * @brief In-place element-wise tensor multiply
 * 
 * @tparam T 
 * @param tensor1 tensor1 
 * @param tensor2 tensor2 
 * @param output_tensor Output
 */
template<typename T>
void TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2, const std::shared_ptr<Tensor<T>>& output_tensor);

/**
 * @brief Clones a tensor
 * 
 * @tparam T 
 * @param tensor Tensor to clone 
 * @return Deep copy of tensor 
 */
template<typename T>
std::shared_ptr<Tensor<T>> TensorClone(std::shared_ptr<Tensor<T>> tensor);

/**
 * @brief Broadcasts two tensors
 * 
 * @tparam T 
 * @param tensor1 tensor1 
 * @param tensor2 tensor2 
 * @return Tensors with same shape 
 */
template<typename T>
std::tuple<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> TensorBroadcast(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2);

}  // shukainfer
#endif // SHUKAINFER_DATA_TENSORUTIL_HPP