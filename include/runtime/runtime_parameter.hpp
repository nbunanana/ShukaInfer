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
        explicit RuntimeParameter(RuntimeParameterType type) : type(type) {}
        RuntimeParameterType type = RuntimeParameterType::KParameterUnknown;
    };

    struct RuntimeParameterInt : public RuntimeParameter {
        explicit RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::KParameterInt) {}
        explicit RuntimeParameterInt(int32_t param_value) :RuntimeParameter(RuntimeParameterType::KParameterInt), value(param_value) {}
        int32_t value = 0;
    };

    struct RuntimeParameterFloat : public RuntimeParameter {
        explicit RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::KParameterFloat) {}
        explicit RuntimeParameterFloat(float param_value) :RuntimeParameter(RuntimeParameterType::KParameterFloat), value(param_value) {}
        float value = 0.f;
    };

    struct RuntimeParameterString : public RuntimeParameter {
        explicit RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::KParameterString) {}
        explicit RuntimeParameterString(std::string param_value) :RuntimeParameter(RuntimeParameterType::KParameterString), value(std::move(param_value)) {}
        std::string value;
    };

    struct RuntimeParameterBool : public RuntimeParameter {
        explicit RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::KParameterBool) {}
        explicit RuntimeParameterBool(bool param_value) :RuntimeParameter(RuntimeParameterType::KParameterBool), value(param_value) {}
        bool value = false;
    };

    struct RuntimeParameterIntArray : public RuntimeParameter {
        explicit RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::KParameterIntArray) {}
        explicit RuntimeParameterIntArray(std::vector<int32_t> param_value) :RuntimeParameter(RuntimeParameterType::KParameterIntArray), value(std::move(param_value)) {}
        std::vector<int32_t> value;
    };

    struct RuntimeParameterStringArray : public RuntimeParameter {
        explicit RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::KParameterStringArray) {}
        explicit RuntimeParameterStringArray(std::vector<std::string> param_value) :RuntimeParameter(RuntimeParameterType::KParameterStringArray), value(std::move(param_value)) {}
        std::vector<std::string> value;
    };

    struct RuntimeParameterFloatArray : public RuntimeParameter {
        explicit RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::KParameterFloatArray) {}
        explicit RuntimeParameterFloatArray(std::vector<float> param_value) :RuntimeParameter(RuntimeParameterType::KParameterFloatArray), value(std::move(param_value)) {}
        std::vector<float> value;
    };

}// shukainfer
#endif //SHUKAINFER_RUNTIME_RUNTIMEPARAMETER_HPP
