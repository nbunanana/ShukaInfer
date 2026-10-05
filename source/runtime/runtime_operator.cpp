/*
 * @file    runtime_operator.cpp
 * @author  nbunanana
 * @date    10/5/26
 * @brief   ${DO_NOT_EDIT}
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "runtime/runtime_operator.hpp"

namespace shukainfer {

    template<typename T>
    bool RuntimeOperatorBase<T>::has_attribute(const std::string &attribute_name) {
        if (attribute_name.empty()) {
            return false;
        }
        if (this->attributes.find(attribute_name) == this->attributes.end()) {
            return false;
        }
        return true;
    }

    template<typename T>
    bool RuntimeOperatorBase<T>::has_parameter(const std::string &parameter_name) {
        if (parameter_name.empty()) {
            return false;
        }
        if (this->attributes.find(parameter_name) == this->attributes.end()) {
            return false;
        }
        return true;
    }

    void RuntimeOperatorUtils<float>::InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator> > &input_operators) {

    }

    void RuntimeOperatorUtils<float>::InitOperatorOutput(const std::vector<pnnx::Operator *> &pnnx_operators,
                                                         const std::vector<std::shared_ptr<RuntimeOperator> > &output_operators) {

    }
} //shukainfer
