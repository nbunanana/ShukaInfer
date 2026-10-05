/*
 * @file    code_status.hpp
 * @author  nbunanana
 * @date    9/27/26
 * @brief   Enumeration type describing parameters type and program state.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_CODESTATUS_HPP
#define SHUKAINFER_CODESTATUS_HPP

namespace shukainfer {
    /**
     * @brief Enumeration type describing parameters type.
     */
    enum class RuntimeParameterType {
        KParameterUnknown = 0,
        KParameterBool = 1,

        KParameterInt = 2,
        KParameterFloat = 3,
        KParameterString = 4,

        KParameterIntArray = 5,
        KParameterFloatArray = 6,
        KParameterStringArray = 7,
    };

    /**
     * @brief  Enumeration type describing program state.
     */
    enum class StatusCode {
        KUnknowCode = -1,
        KSuccess = 0,

        KInferInputEmpty = 1,
        KInferOutputEmpty = 2,
        KInferParamError = 3,
        KInferDimMismatch = 4,

        KFunctionNotImplement = 5,
        KParseWeightError = 6,
        KParseParamError = 7,
        KParseNullOperator = 8,
    };
} // shukainfer
#endif //SHUKAINFER_CODESTATUS_HPP
