/*
 * @file    runtime_attribute.hpp
 * @author  nbunanana
 * @date    9/27/26
 * @brief   Runtime operator attribute.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_RUNTIME_RUNTIMEATTRIBUTE_HPP
#define SHUKAINFER_RUNTIME_RUNTIMEATTRIBUTE_HPP
#include <glog/logging.h>
#include <vector>
#include "runtime_datatype.hpp"
#include "code_status.hpp"

namespace shukainfer {
    /**
     * @brief Runtime operator attribute.
     *
     * Represents an attribute like weights or biases for a graph operator.
     * Contains the attribute data, shape, and data type.
     */
    struct  RuntimeAttribute {

        RuntimeAttribute() = default;

        explicit RuntimeAttribute(std::vector<int32_t> shapes, RuntimeDataType type, std::vector<char> weight_data)
            : weight_data(std::move(weight_data)), shapes(std::move(shapes)), type(type) {}

        /// Attribute data(Typically contains the binary weight values.).
        std::vector<char> weight_data;

        /// Shape of the attribute.
        std::vector<int32_t> shapes;

        /// Data type of the attribute.
        RuntimeDataType type = RuntimeDataType::kTypeUnknow;

        /**
         * @brief Gets the attribute data as a typed array.
         *
         * Returns the weight data as a vector of the template type T.
         * The attribute data is cleared after get by default.
         *
         * @tparam T Data type to return (float, int, etc.).
         * @param need_clear_weights Whether to clear data after get.
         * @return Vector containing the attribute data.
         */
        template<typename T>
        std::vector<T> get(bool need_clear_weights = true);
    };
} // shukainfer
#endif //SHUKAINFER_RUNTIME_RUNTIMEATTRIBUTE_HPP
