/*
 * @file    runtime_parameter.hpp
 * @author  nbunanana
 * @date    9/28/26
 * @brief   The parameters of operator
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_RUNTIME_RUNTIMEPARAMETER_HPP
#define SHUKAINFER_RUNTIME_RUNTIMEPARAMETER_HPP

#include <vector>
#include <string>
#include "code_status.hpp"

/**
 * @brief Runtime data types for operator parameters like scalars and hyperparameters.
 *
 * The parameters of operator have 7 type:
 *   1.int;
 *   2.float
 *   3.string
 *   4.bool
 *   5.int array
 *   6.string array
 *   7.float array
 */
namespace shukainfer {
    struct RuntimeParameter {
        virtual ~RuntimeParameter() = default;
        explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::kParameterUnknown) : type(type) {}
        RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
    };

    struct RuntimeParameterInt : public RuntimeParameter {
        explicit RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::kParameterInt) {}
        explicit RuntimeParameterInt(int32_t param_value) :RuntimeParameter(RuntimeParameterType::kParameterInt), value(param_value) {}
        int32_t value = 0;
    };

    struct RuntimeParameterFloat : public RuntimeParameter {
        explicit RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::kParameterFloat) {}
        explicit RuntimeParameterFloat(float param_value) :RuntimeParameter(RuntimeParameterType::kParameterFloat), value(param_value) {}
        float value = 0.f;
    };

    struct RuntimeParameterString : public RuntimeParameter {
        explicit RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::kParameterString) {}
        explicit RuntimeParameterString(std::string param_value) :RuntimeParameter(RuntimeParameterType::kParameterString), value(std::move(param_value)) {}
        std::string value;
    };

    struct RuntimeParameterBool : public RuntimeParameter {
        explicit RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::kParameterBool) {}
        explicit RuntimeParameterBool(bool param_value) :RuntimeParameter(RuntimeParameterType::kParameterBool), value(param_value) {}
        bool value = false;
    };

    struct RuntimeParameterIntArray : public RuntimeParameter {
        explicit RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {}
        explicit RuntimeParameterIntArray(std::vector<int32_t> param_value) :RuntimeParameter(RuntimeParameterType::kParameterIntArray), value(std::move(param_value)) {}
        std::vector<int32_t> value;
    };

    struct RuntimeParameterStringArray : public RuntimeParameter {
        explicit RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {}
        explicit RuntimeParameterStringArray(std::vector<std::string> param_value) :RuntimeParameter(RuntimeParameterType::kParameterStringArray), value(std::move(param_value)) {}
        std::vector<std::string> value;
    };

    struct RuntimeParameterFloatArray : public RuntimeParameter {
        explicit RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {}
        explicit RuntimeParameterFloatArray(std::vector<float> param_value) :RuntimeParameter(RuntimeParameterType::kParameterFloatArray), value(std::move(param_value)) {}
        std::vector<float> value;
    };

}// shukainfer
#endif //SHUKAINFER_RUNTIME_RUNTIMEPARAMETER_HPP
