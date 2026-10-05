/*
 * @file   tensor.cpp
 * @author Dyc
 * @date   2026-09-16
 * @brief  This is Tensor head code
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#ifndef SHUKAINFER_DATA_TENSOR_HPP
#define SHUKAINFER_DATA_TENSOR_HPP
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>

namespace shukainfer {
template<typename T>
class Tensor {
 public:
    explicit Tensor(T* raw_ptr, uint32_t size);

    explicit Tensor(T* raw_ptr, uint32_t rows, uint32_t cols);

    explicit Tensor(T* raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols);

    explicit Tensor(T* raw_ptr, const std::vector<uint32_t>& shapes);

    /**
     * @brief Construct a empty Tensor(default constructor)
     * 
     */
    explicit Tensor() = default;

    /**
     * @brief Construct a new 3D Tensor 
     * 
     * @param channels Number of channels
     * @param rows Number of rows
     * @param cols Number of cols
     */
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

    /**
     * @brief Construct a new 2D Tensor 
     * 
     * @param rows Number of rows
     * @param cols Number of cols
     */
    explicit Tensor(uint32_t rows, uint32_t cols);

    /**
     * @brief Construct a new 1D Tensor 
     * 
     * @param size Number of vector
     */
    explicit Tensor(uint32_t size);

    /**
     * @brief Construct a new Tensor with shapes
     * 
     * @param shapes Tensor dimensions(维度)
     */
    explicit Tensor(const std::vector<uint32_t>& shapes);

    /**
     * @brief Gets number of rows
     * 
     * @return Number of rows 
     */
    uint32_t rows() const;

    /**
     * @brief Gets number of cols
     * 
     * @return Number of cols 
     */
    uint32_t cols() const;

    /**
     * @brief Gets number of channels
     * 
     * @return Number of channels
     */
    uint32_t channels() const;

    /**
     * @brief Gets total number of elements
     * 
     * @return Total number of elements
     */
    size_t size() const;

    /**
     * @brief return size with matrix of each tesor
     * 
     * @return size_t 
     */
    size_t plane_size() const;


    /**
     * @brief Set the tensor data 
     * 
     * @param data Data to set
     */
    void set_data(const arma::Cube<T>& data);

    /**
     * @brief Determine whether a tensor is empty
     * 
     * @return true tensor is empty
     * @return false tensor is not empty
     */
    bool empty() const;

    /**
     * @brief Gets element reference at offset
     * 
     * @param offset Offset relative to the first address of the tensor
     * @return T& 
     */
    T& index(uint32_t offset);

    /**
     * @brief Gets element reference at offset
     * 
     * @param offset Offset relative to the first address of the tensor
     * @return const T 
     */
    const T index(uint32_t offset) const;

    /**
     * @brief Get tensor shape
     * 
     * @return Tensor dimensions
     */
    const std::vector<uint32_t> shapes() const;

    /**
     * @brief Gets raw tensor shape(raw_shapes_)
     * 
     * @return Raw tensor dimensions
     */
    const std::vector<uint32_t> raw_shapes() const;

    /**
     * @brief Get tensor data
     * 
     * @return Tensor data
     */
    arma::Cube<T>& data();

    /**
     * @brief Gets tensor data (const)
     * 
     * @return Tensor data
     */
    const arma::Cube<T>& data() const;

    /**
     * @brief Get channel matrix
     * 
     * @param channel Channel index
     * @return Channel matrix
     */
    arma::Mat<T>& slice(uint32_t channel);

    /**
     * @brief Get channel matrix(const)
     * 
     * @param channel Channel index
     * @return Channel matrix
     */
    const arma::Mat<T>& slice(uint32_t channel) const;

    /**
     * @brief Gets element at location
     * 
     * @param channel Channel
     * @param row  Row index
     * @param col  col index
     * @return Element at location 
     */
    T& at(uint32_t channel, uint32_t row, uint32_t col);

    /**
     * @brief Gets element at location(const)
     * 
     * @param channel Channel
     * @param row  Row index
     * @param col  col index
     * @return Element at location 
     */
    const T at(uint32_t channel, uint32_t row, uint32_t col) const;

    /**
     * @brief Pads the tensor
     * 
     * @param pads  Padding amount for dimensions
     * @param padding_values Padding value
     */
    void Padding(const std::vector<uint32_t> pads, T padding_values);

    /**
     * @brief Fill tensor with value(cover)
     * 
     * @param value Fill value
     */
    void Fill(T value);

    /**
     * @brief Fills tensor with vector data
     * 
     * @param values Data to fill
     * @param row_major Is it padded according to row_major
     */
    void Fill(const std::vector<T>& values, bool row_major = true);

    /**
     * @brief Gets tensor data as vector
     * 
     * @param row_major Is it padded according to row_major
     * @return tensor data as vector 
     */
    std::vector<T> values(bool row_major = true);

    /**
     * @brief Initializes with 1(T)
     * 
     */
    void Ones();

    /**
     * @brief Initializes with normal distribution
     * 
     * @param mean Mean value
     * @param Variance Variance value
     */
    void RandN(T mean = 0, T Variance = 1);

    /**
     * @brief Initializes with uniform distribution
     * 
     * @param min Minmum value
     * @param max Maxmum value
     */
    void RandU(T min = 0, T max = 1);

    /**
     * @brief Prints tensor
     * 
     */
    void Show();

    /**
     * @brief Reshape tensor
     * 
     * @param shapes New shape
     * @param row_major Is it padded according to row_major
     */
    void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);

    /**
     * @brief Flattens the tensor
     * 
     * @param raw_major Is it padded according to row_major
     */
    void Flatten(bool raw_major = true);

    /**
     * @brief Applies element-wise transform
     * 
     * @param filter Transform function
     */
    void Transform(const std::function<T(T)>& filter);

    /**
     * @brief Gets raw ptr pointer
     * 
     * @return Raw tensor pointer 
     */
    T* raw_ptr();

    /**
     * @brief Gets raw ptr pointer
     * 
     * @return Raw tensor pointer 
     */
    const T* raw_ptr() const;

    /**
     * @brief Gets raw data pointer with offset
     * 
     * @param offset 
     * @return Raw tensor pointer + offset 
     */
    T* raw_ptr(size_t offset);

    /**
     * @brief Gets raw data pointer with offset
     * 
     * @param offset 
     * @return Raw tensor pointer + offset 
     */
    const T* raw_ptr(size_t offset) const;

    /**
     * @brief Gets raw tensor pointer
     * 
     * @param index 
     * @return Raw tensor pointer 
     */
    T* matrix_raw_ptr(uint32_t index);

    const T* matrix_raw_ptr(uint32_t index) const;
 
 private:
    /**
     * @brief The reshape method in the row-major order
     * 
     * @param shapes Tensor shape
     */
    void Review_(const std::vector<uint32_t>& shapes);

    // Raw tensor dimensions
    std::vector<uint32_t> raw_shapes_;

    // Tensor data
    arma::Cube<T> data_;
    
};

template <typename T = float>
using sptr_tensor = std::shared_ptr<Tensor<T>>;

using ftensor = Tensor<float>;
using sptr_ftensor = std::shared_ptr<Tensor<float>>;

using utensor = Tensor<uint8_t>;
using sptr_utensor = std::shared_ptr<Tensor<uint8_t>>;

} // shukainfer

#endif // SHUKAINFER_DATA_TENSOR_HPP