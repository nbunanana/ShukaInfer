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
        kTypeUnknow = 0,
        kTypeFloat16 = 1,
        kTypeFloat32 = 2,
        kTypeFloat64 = 3,
        kTypeInt8 = 4,
        kTypeUInt8 = 5,
        kTypeInt16 = 6,
        kTypeUInt32 = 7,
        kTypeInt64 = 8,
    };
}// shukainfer
#endif //SHUKAINFER_RUNTIME_RUNTIMEDATATYPE_HPP
