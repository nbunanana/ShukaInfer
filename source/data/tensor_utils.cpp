/*
 * @file   tensor_util.cpp
 * @author Dyc
 * @date   2026-09-23
 * @brief
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include <algorithm>
#include <cstdint>
#include <glog/logging.h>
#include <memory>
#include <tuple>
#include "data/tensor.hpp"
#include "data/tensor_utils.hpp"

namespace shukainfer {
    template<typename T>
    std::shared_ptr<Tensor<T>> TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols) {
        return std::make_shared<Tensor<T>>(channels, rows, cols);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> TensorCreate(uint32_t rows, uint32_t cols) {
        return std::make_shared<Tensor<T>>(1,rows, cols);
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> TensorCreate(uint32_t size) {
        return std::make_shared<Tensor<T>>(1,1,size);
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> TensorCreate(const std::vector<uint32_t>& shapes) {
    CHECK(!shapes.empty() && shapes.size() <= 3);
    if (shapes.size() == 1) {
        return std::make_shared<Tensor<T>>(shapes.at(0));
    } else if (shapes.size() == 2) {
        return std::make_shared<Tensor<T>>(shapes.at(0), shapes.at(1));
    } else {
        return std::make_shared<Tensor<T>>(shapes.at(0), shapes.at(1), shapes.at(2));
    }
    }
    

    template<typename T>
    bool TensorIsSame(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2, T threshold) {
        CHECK_NE(tensor1, nullptr);
        CHECK_NE(tensor2, nullptr);
        if(tensor1->shapes() != tensor2->shapes()) {
            return false;
        }
        bool is_same = arma::approx_equal(tensor1->data(), tensor2->data(), "absdiff", threshold);
        return is_same;
    }

    template<typename T>
    std::shared_ptr<Tensor<T>> TensorPadding(const std::shared_ptr<Tensor<T>> tensor, const std::vector<uint32_t>& pads, T padding_value) {
        /* Theoretically, the Padding method in the Tensor class can be directly invoked here. However, for the sake of completeness, 
         I have written a more low-level method. The performance comparison between this method and the subcube function of Cube is still unclear. */
        CHECK_NE(tensor, nullptr) << "&tensor is nullptr!!";
        CHECK(!tensor->empty()) << "The data area of the tensor is empty.";
        CHECK_EQ(pads.size(), 4) << "The size of pads isn't 4!!";
        /* tensor->Padding(pads, padding_value);
        return tensor; */
        uint32_t up = pads.at(0);
        uint32_t down = pads.at(1);
        uint32_t left = pads.at(2);
        uint32_t right = pads.at(3);

        std::shared_ptr<Tensor<T>> output = std::make_shared<Tensor<T>>(tensor->channels(), tensor->rows() + up + down, tensor->cols() + left + right);
        for(uint32_t channel = 0; channel < tensor->channels(); channel++) {
            const arma::Mat<T>& in_channel = tensor->slice(channel);
            arma::Mat<T>& out_channel= output->slice(channel);

            for(uint32_t col = 0; col < tensor->cols(); col++) {
                T* out_channel_ptr = out_channel.colptr(col + left);
                const T* in_channel_ptr = in_channel.colptr(col);

                // Copy the data from the original cube to the corresponding position in the new cube.
                for(uint32_t row = 0; row < tensor->rows(); row++) {
                    const T value = *(in_channel_ptr + row);
                    *(out_channel_ptr + row + up) = value;
                }

                // Fill the up and down areas with padding_value.
                for(uint32_t row = 0; row < up; row++) {
                    *(out_channel_ptr + row) = padding_value;
                }
                for(uint32_t row = 0; row < down; row++) {
                    *(out_channel_ptr + tensor->rows() + up + row) = padding_value;
                }
            }

            // Fill the left areas with padding_value.
            for(uint32_t col = 0; col < left; col++) {
                T* out_channel_ptr = out_channel.colptr(col);
                for(uint32_t row = 0; row < tensor->rows() + up + down; row++) {
                    *(out_channel_ptr + row) = padding_value;
                }
            }

            // Fill the right areas with padding_value.
            for(uint32_t col = 0; col < right; col++) {
                T* out_channel_ptr = out_channel.colptr(left + tensor->cols() + col);
                for(uint32_t row = 0; row < tensor->rows() + up + down; row++) {
                    *(out_channel_ptr + row) = padding_value;
                }
            }
        }
        return output;
    }

template<typename T>
std::tuple<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> TensorBroadcast(const std::shared_ptr<Tensor<T>> &tensor1, const std::shared_ptr<Tensor<T>>& tensor2) {
    /*
        NOTE: The broadcast function only adapted (1x1xc and rxwxc) to (rxwxc and rxwxc).    Fill previous vaule with per-channel to new tensor with per-channel!!
    */
    CHECK_NE(tensor1, nullptr) << "&tensor1 is nullptr!!";
    CHECK_NE(tensor2, nullptr) << "&tensor2 is nullptr!!"; 
    if(tensor1->shapes() == tensor2->shapes()) {
        return {tensor1, tensor2};
    } else {
        CHECK_EQ(tensor1->channels(), tensor2->channels());
        if(tensor1->rows() == 1 && tensor1->cols() == 1) {
            std::shared_ptr<Tensor<T>> new_tensor = TensorCreate<T>(tensor2->channels(), tensor2->rows(), tensor2->cols());
            for(uint32_t channel = 0; channel < tensor1->channels(); channel++) {
                T* new_tensor_ptr = new_tensor->matrix_raw_ptr(channel);
                std::fill(new_tensor_ptr, new_tensor_ptr + new_tensor->plane_size(), tensor1->index(channel));
            }
            return {new_tensor, tensor2};
        } else if(tensor2->rows() == 1 && tensor2->cols() == 1) {
            std::shared_ptr<Tensor<T>> new_tensor = TensorCreate<T>(tensor1->channels(), tensor1->rows(), tensor1->cols());
            for(uint32_t channel = 0; channel < tensor2->channels(); channel++) {
                T* new_tensor_ptr = new_tensor->matrix_raw_ptr(channel);
                std::fill(new_tensor_ptr, new_tensor_ptr + new_tensor->plane_size(), tensor2->index(channel));
            }
            return {tensor1, new_tensor};
        } else {
            LOG(FATAL) << "Broadcast shape is not adapting!";
            return {tensor1, tensor2};
        }
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> TensorClone(std::shared_ptr<Tensor<T>> tensor) {
    return std::make_shared<Tensor<T>>(*tensor);
}

template<typename T>
std::shared_ptr<Tensor<T>> TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2) {
    CHECK_NE(tensor1, nullptr) << "&tensor1 is nullptr!!";
    CHECK_NE(tensor2, nullptr) << "&tensor2 is nullptr!!"; 
    if(tensor1->shapes() == tensor2->shapes()) {
        std::shared_ptr<Tensor<T>> out_tensor = TensorCreate<T>(tensor1->channels(), tensor1->rows(), tensor1->cols());
        out_tensor->set_data(tensor1->data() + tensor2->data());
        return out_tensor;
    } else {
        CHECK_EQ(tensor1->channels(), tensor2->channels()) << "The channels of tensor1 and tensor2 are not the same, so broadcasting cannot be performed!!";
        const auto& [broadcast_tensor1, broadcast_tensor2] = TensorBroadcast(tensor1, tensor2);
        CHECK(broadcast_tensor1->shapes() == broadcast_tensor2->shapes());
        std::shared_ptr<Tensor<T>> out_tensor = TensorCreate<T>(broadcast_tensor1->channels(), broadcast_tensor1->rows(), broadcast_tensor1->cols());
        out_tensor->set_data(broadcast_tensor1->data() + broadcast_tensor2->data());
        return out_tensor;
    }
}

template<typename T>
void TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2, const std::shared_ptr<Tensor<T>>& output_tensor) {
    CHECK_NE(tensor1, nullptr) << "&tensor1 is nullptr!!";
    CHECK_NE(tensor2, nullptr) << "&tensor2 is nullptr!!"; 
    if(tensor1->shapes() == tensor2->shapes()) {
        CHECK(output_tensor->shapes() ==  tensor1->shapes());
        output_tensor->set_data(tensor1->data() + tensor2->data());
    } else {
        CHECK_EQ(tensor1->channels(), tensor2->channels()) << "The channels of tensor1 and tensor2 are not the same, so broadcasting cannot be performed!!";
        const auto& [broadcast_tensor1, broadcast_tensor2] = TensorBroadcast(tensor1, tensor2);
        CHECK(output_tensor->shapes() == broadcast_tensor1->shapes());
        CHECK(output_tensor->shapes() == broadcast_tensor2->shapes());
        output_tensor->set_data(broadcast_tensor1->data() + broadcast_tensor2->data());
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2) {
    CHECK_NE(tensor1, nullptr) << "&tensor1 is nullptr!!";
    CHECK_NE(tensor2, nullptr) << "&tensor2 is nullptr!!"; 
    if(tensor1->shapes() == tensor2->shapes()) {
        std::shared_ptr<Tensor<T>> out_tensor = TensorCreate<T>(tensor1->shapes());
        out_tensor->set_data(tensor1->data() % tensor2->data());
        return out_tensor;
    } else {
        CHECK_EQ(tensor1->channels(), tensor2->channels()) << "The channels of tensor1 and tensor2 are not the same, so broadcasting cannot be performed!!";
        const auto& [broadcast_tensor1, broadcast_tensor2] = TensorBroadcast(tensor1, tensor2);
        CHECK(broadcast_tensor1->shapes() == broadcast_tensor2->shapes());
        std::shared_ptr<Tensor<T>> out_tensor = TensorCreate<T>(broadcast_tensor1->shapes());
        out_tensor->set_data(broadcast_tensor1->data() % broadcast_tensor2->data());
        return out_tensor;
    }
}

template<typename T>
void TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2, const std::shared_ptr<Tensor<T>>& output_tensor) {
    CHECK_NE(tensor1, nullptr) << "&tensor1 is nullptr!!";
    CHECK_NE(tensor2, nullptr) << "&tensor2 is nullptr!!"; 
    if(tensor1->shapes() == tensor2->shapes()) {
        CHECK(output_tensor->shapes() == tensor1->shapes());
        output_tensor->set_data(tensor1->data() % tensor2->data());
    } else {
        CHECK_EQ(tensor1->channels(), tensor2->channels()) << "The channels of tensor1 and tensor2 are not the same, so broadcasting cannot be performed!!";
        const auto& [broadcast_tensor1, broadcast_tensor2] = TensorBroadcast(tensor1, tensor2);
        CHECK(output_tensor->shapes() == broadcast_tensor1->shapes());
        CHECK(output_tensor->shapes() == broadcast_tensor2->shapes());
        output_tensor->set_data(broadcast_tensor1->data() % broadcast_tensor2->data());
    }
}


    // ------------------ float 实例化 ------------------
template std::shared_ptr<Tensor<float>>
TensorCreate<float>(uint32_t channels, uint32_t rows, uint32_t cols);

template std::shared_ptr<Tensor<float>>
TensorCreate<float>(uint32_t rows, uint32_t cols);

template std::shared_ptr<Tensor<float>>
TensorCreate<float>(uint32_t size);

template std::shared_ptr<Tensor<float>>
TensorCreate<float>(const std::vector<uint32_t>& shapes);

template bool
TensorIsSame<float>(const std::shared_ptr<Tensor<float>>& tensor1,
                    const std::shared_ptr<Tensor<float>>& tensor2, float threshold);

template std::shared_ptr<Tensor<float>>
TensorPadding<float>(const std::shared_ptr<Tensor<float>> tensor,
                     const std::vector<uint32_t>& pads, float padding_value);

template std::tuple<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>>
TensorBroadcast<float>(const std::shared_ptr<Tensor<float>>& tensor1,
                       const std::shared_ptr<Tensor<float>>& tensor2);

template std::shared_ptr<Tensor<float>>
TensorClone<float>(std::shared_ptr<Tensor<float>> tensor);

template std::shared_ptr<Tensor<float>>
TensorElementAdd<float>(const std::shared_ptr<Tensor<float>>& tensor1,
                        const std::shared_ptr<Tensor<float>>& tensor2);

template void
TensorElementAdd<float>(const std::shared_ptr<Tensor<float>>& tensor1,
                        const std::shared_ptr<Tensor<float>>& tensor2,
                        const std::shared_ptr<Tensor<float>>& output_tensor);

template std::shared_ptr<Tensor<float>>
TensorElementMultiply<float>(const std::shared_ptr<Tensor<float>>& tensor1,
                             const std::shared_ptr<Tensor<float>>& tensor2);

template void
TensorElementMultiply<float>(const std::shared_ptr<Tensor<float>>& tensor1,
                             const std::shared_ptr<Tensor<float>>& tensor2,
                             const std::shared_ptr<Tensor<float>>& output_tensor);

// ------------------ uint8_t 实例化 ------------------
template std::shared_ptr<Tensor<uint8_t>>TensorCreate<uint8_t>(uint32_t channels, uint32_t rows, uint32_t cols);

template std::shared_ptr<Tensor<uint8_t>>TensorCreate<uint8_t>(uint32_t rows, uint32_t cols);

template std::shared_ptr<Tensor<uint8_t>>TensorCreate<uint8_t>(uint32_t size);

template std::shared_ptr<Tensor<uint8_t>>TensorCreate<uint8_t>(const std::vector<uint32_t>& shapes);

template bool TensorIsSame<uint8_t>(const std::shared_ptr<Tensor<uint8_t>>& tensor1,
                      const std::shared_ptr<Tensor<uint8_t>>& tensor2, uint8_t threshold);

template std::shared_ptr<Tensor<uint8_t>>TensorPadding<uint8_t>(const std::shared_ptr<Tensor<uint8_t>> tensor,
                       const std::vector<uint32_t>& pads, uint8_t padding_value);

template std::tuple<std::shared_ptr<Tensor<uint8_t>>, std::shared_ptr<Tensor<uint8_t>>>TensorBroadcast<uint8_t>(const std::shared_ptr<Tensor<uint8_t>>& tensor1,
                         const std::shared_ptr<Tensor<uint8_t>>& tensor2);

template std::shared_ptr<Tensor<uint8_t>>TensorClone<uint8_t>(std::shared_ptr<Tensor<uint8_t>> tensor);

template std::shared_ptr<Tensor<uint8_t>>TensorElementAdd<uint8_t>(const std::shared_ptr<Tensor<uint8_t>>& tensor1,
                          const std::shared_ptr<Tensor<uint8_t>>& tensor2);

template void TensorElementAdd<uint8_t>(const std::shared_ptr<Tensor<uint8_t>>& tensor1,
                          const std::shared_ptr<Tensor<uint8_t>>& tensor2,
                          const std::shared_ptr<Tensor<uint8_t>>& output_tensor);

template std::shared_ptr<Tensor<uint8_t>>TensorElementMultiply<uint8_t>(const std::shared_ptr<Tensor<uint8_t>>& tensor1,
                               const std::shared_ptr<Tensor<uint8_t>>& tensor2);

template void TensorElementMultiply<uint8_t>(const std::shared_ptr<Tensor<uint8_t>>& tensor1,
                               const std::shared_ptr<Tensor<uint8_t>>& tensor2,
                               const std::shared_ptr<Tensor<uint8_t>>& output_tensor);

} // shukainfer
