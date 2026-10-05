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

    enum class RuntimeParameterType {
        kParameterUnknown = 0,
        kParameterBool = 1,
        kParameterInt = 2,

        kParameterFloat = 3,
        kParameterString = 4,
        kParameterIntArray = 5,
        kParameterFloatArray = 6,
        kParameterStringArray = 7,
      };

    enum class StatusCode {
        kUnknownCode = -1,
        kSuccess = 0,

        kInferInputsEmpty = 1,
        kInferOutputsEmpty = 2,
        kInferParamError = 3,
        kInferDimMismatch = 4,

        kFunctionNotImplement = 5,
        kParseWeightError = 6,
        kParseParamError = 7,
        kParseNullOperator = 8,
      };

}  // shukainfer
#endif  // SHUKAINFER_CODESTATUS_HPP
