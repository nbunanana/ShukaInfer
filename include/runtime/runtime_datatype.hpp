/*
 * @file    runtime_datatype.hpp
 * @author  nbunanana
 * @date    9/27/26
 * @brief   Runtime data types for operator attributes.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_RUNTIME_RUNTIMEDATATYPE_HPP
#define SHUKAINFER_RUNTIME_RUNTIMEDATATYPE_HPP
namespace shukainfer {
    /**
     * @brief Runtime data types for operator attributes.
     *
     * Enumerates the data types supported for operator attributes like weights and biases.
     */
    enum class RuntimeDataType {
        KTypeUnknow = 0,
        KTypeFloat16 = 1,
        KTypeFloat32 = 2,
        KTypeFloat64 = 3,
        KTypeInt8 = 4,
        KTypeUInt8 = 5,
        KTypeInt16 = 6,
        KTypeUInt32 = 7,
        KTypeInt64 = 8,
    };
}// shukainfer
#endif //SHUKAINFER_RUNTIME_RUNTIMEDATATYPE_HPP
