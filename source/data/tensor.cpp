/*
 * @file   tensor.cpp
 * @author Dyc
 * @date   2026-09-17
 * @brief This is Tensor source code
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */


#include "data/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glog/logging.h>
#include <iterator>
#include <numeric>
#include <random>
#include <strings.h>
#include <utility>
#include <vector>

namespace shukainfer {

template <typename T>
Tensor<T>::Tensor(T* raw_ptr, uint32_t size) {
   CHECK_NE(raw_ptr, nullptr);
   this->data_ = arma::Cube<T>(raw_ptr, 1, size, 1, false, true);
   this->raw_shapes_ = std::vector<uint32_t>{size};
}

template <typename T>
Tensor<T>::Tensor(T* raw_ptr, uint32_t rows, uint32_t cols) {
   CHECK_NE(raw_ptr, nullptr);
   this->data_ = arma::Cube<T>(raw_ptr, rows, cols, 1, false, true);
   if (rows == 1) {
      this->raw_shapes_ = std::vector<uint32_t>{cols};
   } else {
      this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
   }
}

template <typename T>
Tensor<T>::Tensor(T* raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols) {
   CHECK_NE(raw_ptr, nullptr);
   this->data_ = arma::Cube<T>(raw_ptr, rows, cols, channels, false, true);
   if (channels == 1 && rows == 1) {
      this->raw_shapes_ = std::vector<uint32_t>{cols};
   } else if (channels == 1) {
      this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
   } else {
      this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
   }
}

template<typename T>
Tensor<T>::Tensor(T* raw_ptr, const std::vector<uint32_t>& shapes) {
   CHECK_NE(raw_ptr, nullptr);
   CHECK_EQ(shapes.size(), 3);
   uint32_t channels = shapes.at(0);
   uint32_t rows = shapes.at(1);
   uint32_t cols = shapes.at(2);

   this->data_ = arma::Cube<T>(raw_ptr, rows, cols, channels, false, true);

   if (channels == 1 && rows == 1) {
      this->raw_shapes_ = std::vector<uint32_t>{cols};
   } else if (channels == 1) {
      this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
   } else {
      this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
   }
}

template<typename T>
Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
   this->data_ = arma::Cube<T>(rows, cols, channels);

   if (channels == 1 && rows == 1) {
   this->raw_shapes_ = std::vector<uint32_t>{cols};
   } else if (channels == 1) {
      this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
   } else {
      this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
   }
}

template<typename T>
Tensor<T>::Tensor(uint32_t rows, uint32_t cols) {
   this->data_ = arma::Cube<T>(rows, cols, 1);

   if (rows == 1) {
      this->raw_shapes_ = std::vector<uint32_t>{cols};
   } else {
      this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
   }
}

template<typename T>
Tensor<T>::Tensor(uint32_t size) {
   this->data_ = arma::Cube<T>(1, size, 1);

   this->raw_shapes_ = std::vector<uint32_t>{size};
}

template<typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shapes) {
   CHECK(!shapes.empty() && shapes.size() <= 3);

   uint32_t remaining = 3 - shapes.size();
   std::vector<uint32_t> shapes_(3, 1);
   std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);
   
   uint32_t channels = shapes_.at(0);
   uint32_t rows = shapes_.at(1);
   uint32_t cols = shapes_.at(2);

   this->data_ = arma::Cube<T>(rows ,cols, channels);

      if (channels == 1 && rows == 1) {
   this->raw_shapes_ = std::vector<uint32_t>{cols};
   } else if (channels == 1) {
      this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
   } else {
      this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
   }
}

template<typename T>
uint32_t Tensor<T>::rows() const {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   return this->data_.n_rows;
}

template<typename T>
uint32_t Tensor<T>::cols() const {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   return this->data_.n_cols;
}

template<typename T>
uint32_t Tensor<T>::channels() const {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   return this->data_.n_slices;
}

template<typename T>
size_t Tensor<T>::size() const {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty."; 
   // return this->data_.size() 
   return this->rows() * this->cols() * this->channels();
}

template<typename T>
size_t Tensor<T>::plane_size() const {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty." ;
   return this->cols() * this->rows();
}

template<typename T>
void Tensor<T>::set_data(const arma::Cube<T>& data) {
   CHECK_EQ(this->rows(), data.n_rows) << data.n_rows << "!=" << this->rows();
   CHECK_EQ(this->cols(), data.n_rows) << data.n_cols << "!=" << this->cols();
   CHECK_EQ(this->channels(), data.n_slices) << data.n_slices << "!=" << this->channels();
   this->data_ = data;
}

template<typename T>
bool Tensor<T>::empty() const {
   return this->data_.empty();
}

template<typename T>
T& Tensor<T>::index(uint32_t offset) {
   CHECK_LT(offset, this->size()) << "The index out of bound!";
   return this->data_.at(offset);
}

template<typename T>
const T Tensor<T>::index(uint32_t offset) const {
   CHECK_LT(offset, this->size()) << "The index out of bound!";
   return this->data_.at(offset);
}

template<typename T>
const std::vector<uint32_t> Tensor<T>::shapes() const{
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   return {this->channels(), this->rows(), this->cols()};
}

template<typename T>
const std::vector<uint32_t> Tensor<T>::raw_shapes() const {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   CHECK_LE(this->raw_shapes_.size(), 3);
   CHECK_GE(this->raw_shapes_.size(), 1);
   return this->raw_shapes_;
}

template<typename T>
arma::Cube<T>& Tensor<T>::data() {
   return this->data_;
}

template<typename T>
const arma::Cube<T>& Tensor<T>::data() const {
   return this->data_;
}

template<typename T>
arma::Mat<T>& Tensor<T>::slice(uint32_t channel) {
   CHECK_LT(channel, this->channels()) << "The channel out of bound(more bigger).";
   CHECK_GE(channel, 0) << "The channel out of bound(more smaller).";
   return this->data_.slice(channel);
}

template<typename T>
const arma::Mat<T>& Tensor<T>::slice(uint32_t channel) const {
   CHECK_LT(channel, this->channels()) << "The channel out of bound(more bigger).";
   CHECK_GE(channel, 0) << "The channel out of bound(more smaller).";
   return this->data_.slice(channel);
}

template<typename T>
T& Tensor<T>::at(uint32_t channel , uint32_t row, uint32_t col) {
   CHECK_LT(channel, this->channels()) << "The channel out of bound.";
   CHECK_LT(row, this->rows()) << "The row out of bound.";
   CHECK_LT(col, this->cols()) << "The col out of bound.";
   return this->data_.at(row, col, channel);
}

template<typename T>
const T Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const {
   CHECK_LT(channel, this->channels()) << "The channel out of bound.";
   CHECK_LT(row, this->rows()) << "The row out of bound.";
   CHECK_LT(col, this->cols()) << "The col out of bound.";
   return this->data_.at(row, col, channel);   
}

template<typename T>
void Tensor<T>::Padding(const std::vector<uint32_t> pads, T padding_values) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   CHECK_EQ(pads.size(), 4) << "The pads' size should set 4!";
   uint32_t up = pads.at(0);
   uint32_t down = pads.at(1);
   uint32_t left = pads.at(2);
   uint32_t right = pads.at(3);
   
   arma::Cube<T> new_data_(this->data_.n_rows + up + down, this->data_.n_cols + left + right, this->data_.n_slices);
   new_data_.fill(padding_values);

   // Copy the original data over
   new_data_.subcube(up, left, 0, 
                     new_data_.n_rows - down - 1, new_data_.n_cols - right - 1, new_data_.n_slices - 1) = this->data_;
   
   this->data_ = std::move(new_data_);
   // At this point, data_ has already been filled, and you can directly call the member function
   this->raw_shapes_ = std::vector<uint32_t>{this->channels(), this->rows(), this->cols()};
}

template<typename T>
void Tensor<T>::Fill(T value) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   data_.fill(value);
}

template<typename T>
void Tensor<T>::Fill(const std::vector<T>& values, bool raw_major) {
   CHECK_EQ(this->data_.size(), values.size());
   if(raw_major) {
      const uint32_t rows = this->rows();
      const uint32_t cols = this->cols();
      const uint32_t planes = rows * cols;
      const uint32_t channels = this->channels();

      for(uint32_t i = 0; i < channels; i++) {
         arma::Mat<T> channel_data_(const_cast<T*>(values.data()) + i * planes, this->cols(), this->rows(), false, true);

         this->data_.slice(i) = channel_data_.t();
      }
   } else {
      std::copy(values.begin(), values.end(), this->data_.memptr());
   }
}

template<typename T>
std::vector<T> Tensor<T>::values(bool row_major) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   std::vector<T> values(this->data_.size());

   if(row_major) {
      uint32_t cunt = 0;
      for(uint32_t c = 0l; c < this->channels(); c++) {
         const arma::Mat<T>& channel = this->data_.slice(c).t();
         std::copy(channel.begin(), channel.end(), values.begin() + cunt);
         cunt += channel.size();
      }
      CHECK_EQ(cunt, values.size()) << "An error occurred during the copying process, as the size of values does not equal that of this->data_!!!";
   } else {
      std::copy(this->data_.mem, this->data_.mem + this->data_.size(), values.begin());
   }
   return values;
}

template<typename T>
void Tensor<T>::Ones() {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   this->Fill(T{1});
}

template<>
void Tensor<float>::RandN(float mean, float Variance) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   std::random_device rd;
   std::mt19937 mt(rd());

   std::normal_distribution<float> dist(mean, Variance);
   for(size_t i = 0; i < this->size(); i++) {
      this->index(i) = dist(mt);
   }
}

template <>
void Tensor<int32_t>::RandU(int32_t min, int32_t max) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   std::random_device rd;
   std::mt19937 mt(rd());

   std::uniform_int_distribution<int32_t> dist(min, max);
   for(size_t i = 0; i < this->size(); i++) {
      this->index(i) = dist(mt);
   }
}

template <>
void Tensor<std::uint8_t>::RandU(std::uint8_t min, std::uint8_t max) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   std::random_device rd;
   std::mt19937 mt(rd());

   #ifdef _MSC_VER
      std::uniform_int_distribution<int32_t> dist(0, max - min);
      uint8_t max_value = std::numeric_limits<uint8_t>::max();
      for(uint32_t i = 0; i < this->size(); i++) {
         this->index(i) = min + static_cast<uint8_t>(dist(mt));
      }

   #else
      std::uniform_int_distribution<std::uint8_t> dist(min, max);
      for(size_t i = 0; i < this->size(); i++) {
         this->index(i) = dist(mt);
      }
   #endif
}

template <>
void Tensor<float>::RandU(float min, float max) {
   CHECK(!this-> data_.empty()) << "The data area of the tensor is empty.";
   std::random_device rd;
   std::mt19937 mt(rd());

   std::uniform_real_distribution<float> dist(min, max);
   for(size_t i = 0; i < this->size(); i++) {
      this->index(i) = dist(mt);
   }
}


template<typename T>
void Tensor<T>::Show() {
   if(this->data_.empty()) {
      std::cout << "Tensor is empty!!";
      return;
   }

   for(uint32_t c; c < this->channels(); c++) {
      std::cout << "Channel:" << c;
      std::cout << "\n" << this->data_.slice(c); 
   }
}

template<typename T>
void Tensor<T>::Reshape(const std::vector<uint32_t> &shapes, bool row_major) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   CHECK(!shapes.empty());

   const uint32_t total_new = std::accumulate(shapes.begin(), shapes.end(), static_cast<size_t>(1), std::multiplies<size_t>());
   CHECK_LE(shapes.size(), 3);
   CHECK_EQ(this->size(), total_new);

   if(!row_major) {
      if (shapes.size() == 3) {
         this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
         this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
      } else if (shapes.size() == 2) {
         this->data_.reshape(shapes.at(0), shapes.at(1), 1);
         this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
      } else {
         this->data_.reshape(1,shapes.at(0), 1);
         this->raw_shapes_ = {shapes.at(0)};
      }
   } else {
      if (shapes.size() == 3) {
         this->Review_({shapes.at(0), shapes.at(1), shapes.at(2)});
         this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
      } else if (shapes.size() == 2) {
         this->Review_({1, shapes.at(0), shapes.at(1)});
         this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
      } else {
         this->Review_({1, 1,shapes.at(0)});
         this->raw_shapes_ = {shapes.at(0)};
      }
   }
}

template<typename T>
void Tensor<T>::Flatten(bool raw_major) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   const uint32_t size = this->data_.size();
   this->Reshape({size}, raw_major);
}

template<typename T>
void Tensor<T>::Transform(const std::function<T (T)> &filter) {
   CHECK(!this->empty()) << "The data area of the tensor is empty.";
   this->data_.transform(filter);
}

template<typename T>
T* Tensor<T>::raw_ptr() {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   return this->data_.memptr();
} 

template<typename T>
const T* Tensor<T>::raw_ptr() const {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   return this->data_.memptr();
}

template<typename T>
T* Tensor<T>::raw_ptr(size_t offset) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   CHECK_LT(offset, this->data_.size()) << "Offset out of tensor size!!";
   return this->data_.memptr() + offset;
}

template<typename T>
const T* Tensor<T>::raw_ptr(size_t offset) const {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   CHECK_LT(offset, this->data_.size()) << "Offset out of tensor size!!";
   return this->data_.memptr() + offset;
}

template<typename T>
T* Tensor<T>::matrix_raw_ptr(uint32_t index) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   CHECK_LT(index, this->data_.n_slices);
   size_t offset = index * this->plane_size();
   CHECK_LE(offset, this->data_.size()) << "Offset out of tensor size!!"; 
   T* mem_ptr = this->raw_ptr(offset);
   return mem_ptr;
}

template<typename T>
const T*  Tensor<T>::matrix_raw_ptr(uint32_t index) const {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   CHECK_LT(index, this->data_.n_slices);
   size_t offset = index * this->plane_size();
   CHECK_LE(offset, this->data_.size()) << "Offset out of tensor size!!"; 
   const T* mem_ptr = this->raw_ptr(offset);
   return mem_ptr;
}

template<typename T>
void Tensor<T>::Review_(const std::vector<uint32_t>& shapes) {
   CHECK(!this->data_.empty()) << "The data area of the tensor is empty.";
   CHECK_EQ(shapes.size(), 3);
   
   uint32_t dst_chs = shapes.at(0);
   uint32_t dst_rows = shapes.at(1);
   uint32_t dst_cols = shapes.at(2);

   CHECK_EQ(this->data_.size(), dst_chs * dst_rows * dst_cols);
   arma::Cube<T> new_data(dst_rows, dst_cols, dst_chs);
   const uint32_t dst_plane_size  = dst_rows * dst_cols;

   for(uint32_t ori_channel = 0; ori_channel < this->data_.n_slices; ori_channel++) {
      const uint32_t ori_plane_start = ori_channel * this->data_.n_rows * this->data_.n_cols;
      for(uint32_t ori_col = 0; ori_col < this->data_.n_cols; ori_col++) {
         const T* col_ptr = this->data_.slice_colptr(ori_channel, ori_col);
         for(uint32_t ori_row = 0; ori_row < this->data_.n_rows; ori_row++) {
            const uint32_t dst_pos_idx = ori_plane_start + ori_row * this->data_.n_cols + ori_col;
            const uint32_t dst_ch = dst_pos_idx / dst_plane_size;
            const uint32_t dst_ch_offset = dst_pos_idx % dst_plane_size;
            const uint32_t dst_row = dst_ch_offset / dst_cols;
            const uint32_t dst_col = dst_ch_offset % dst_cols;
            new_data.at(dst_row, dst_col, dst_ch) = *(col_ptr + ori_row);
         }
      }
   }
   this->data_ = std::move(new_data);
}


template class Tensor<float>;
template class Tensor<int32_t>;
template class Tensor<uint8_t>;

} // shukainfer