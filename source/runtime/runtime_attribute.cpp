/*
 * @file    runtime_attribute.cpp
 * @author  nbunanana
 * @date    9/27/26
 * @brief   Runtime operator attribute.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "runtime/runtime_attribute.hpp"

#include <cstring>

namespace shukainfer {
    template<typename T>
    std::vector<T> RuntimeAttribute::get(bool need_clear_weights) {
        CHECK(!this->weight_data.empty());
        CHECK(this->type != RuntimeDataType::KTypeUnknow);
        const uint32_t elem_size = sizeof(T);
        CHECK_EQ(this->weight_data.size() % elem_size, 0) << "Type error!!";
        const uint32_t num_elements = weight_data.size() / elem_size;


        switch (this->type) {
            case RuntimeDataType::KTypeFloat32: {
                CHECK((std::is_same_v<T, float>)) << "Data is float ,but T is not!!";
                break;
            }
            case RuntimeDataType::KTypeInt8:
                CHECK((std::is_same_v<T, int8_t>)) << "Data is int8 ,but T is not!!";
                break;
            default: {
                LOG(FATAL) << "Unknow weight data type: " << static_cast<int32_t>(this->type);
            }
        }

        std::vector<T> outputs(num_elements);
        std::memcpy(outputs.data(), this->weight_data.data(), this->weight_data.size());

        if (need_clear_weights) {
            std::vector<char>().swap(this->weight_data);
        }
        return outputs;
    }

    template std::vector<float> RuntimeAttribute::get(bool);
    template std::vector<int8_t> RuntimeAttribute::get(bool);
} // shukainfer
