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

    void RuntimeOperatorUtils<float>::InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator> > &operators) {
        CHECK(operators.empty()) << "Operators for init input shapes is empty";
        for (const std::shared_ptr<RuntimeOperator>& op : operators) {
            const std::map<std::string, std::shared_ptr<RuntimeOperand>>& input_operand_map = op->input_operands;
            for (const auto& [_, input_operand] : input_operand_map) {
                if (input_operand == nullptr) {
                    continue;
                }

                const RuntimeDataType& type = input_operand->type;
                CHECK(type == RuntimeDataType::kTypeFloat32) << "The graph only support float32 yet!!";

                const std::vector<int32_t> input_operand_shapes = input_operand->shapes;
                auto& input_datas = input_operand->datas;

                CHECK(input_operand_shapes.size() == 2 || input_operand_shapes.size() == 4 ||
                      input_operand_shapes.size() == 3)
                    << "Unsupported tensor shape sizes: " << input_operand_shapes.size();

                const int32_t& batch = input_operand_shapes.at(0);
                CHECK(batch > 0) << "Dynamic batch size is not supported!";

                if (!input_datas.empty()) {
                    CHECK_EQ(input_datas.size(), batch);
                } else {
                    input_datas.resize(batch);
                }
            }
        }
    }

    void RuntimeOperatorUtils<float>::InitOperatorOutput(const std::vector<pnnx::Operator *> &pnnx_operators,
                                                         const std::vector<std::shared_ptr<RuntimeOperator> > &operators) {

    }
} //shukainfer
