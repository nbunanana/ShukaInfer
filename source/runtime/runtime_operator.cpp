/*
 * @file    runtime_operator.cpp
 * @author  nbunanana
 * @date    10/5/26
 * @brief   Base for runtime graph operator
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "runtime/runtime_operator.hpp"
#include  "data/tensor_utils.hpp"

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
        if (operators.empty()) {
            LOG(ERROR) << "operators for init input shapes is  empty!!";
            return;
        }
        for (const auto& op : operators) {
            if (op->input_operands.empty()) {
                continue;
            } else {
                for (const auto& [_, input_operand] : op->input_operands) {
                    if (input_operand == nullptr) {
                        continue;
                    }
                    const RuntimeDataType& type = input_operand->type;
                    auto& input_datas = input_operand->datas;
                    CHECK(type == RuntimeDataType::kTypeFloat32) << "The graph only support float32 yet!!";
                    const auto& input_operand_shapes = input_operand->shapes;

                    CHECK(!input_operand_shapes.empty()) << "input_operand_shapes is empty!!";
                    const int32_t batch = input_operand_shapes.at(0);
                    CHECK(batch >0) << "Dynamic batch size is not supported yet!!";
                    CHECK(input_operand_shapes.size() == 2 || input_operand_shapes.size() == 4 ||
                          input_operand_shapes.size() == 3)
                        << "Unsupported tensor shape sizes: " << input_operand_shapes.size();

                    if (!input_datas.empty()) {
                        CHECK_EQ(input_datas.size(), batch);
                    } else {
                        input_datas.resize(batch);
                    }
                }
            }
        }
    }

    static void CheckAndReshapeTensor(ftensorptr& output_tensor,
                              const std::vector<int32_t>& operand_shapes) {
        switch (operand_shapes.size()) {
            case 4:
                output_tensor->Reshape(
                    {(uint32_t)operand_shapes[1], (uint32_t)operand_shapes[2], (uint32_t)operand_shapes[3]});
                break;
            case 3:
                output_tensor->Reshape({(uint32_t)operand_shapes[1], (uint32_t)operand_shapes[2]});
                break;
            case 2:
                output_tensor->Reshape({(uint32_t)operand_shapes[1]});
                break;
            default:
                LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
                break;
        }
    }

    static ftensorptr CreateTensor(const std::vector<int32_t>& operand_shapes) {
        switch (operand_shapes.size()) {
            case 4:
                return TensorCreate<float>(operand_shapes[1], operand_shapes[2], operand_shapes[3]);
            case 3:
                return TensorCreate<float>(operand_shapes[1], operand_shapes[2]);
            case 2:
                return TensorCreate<float>(operand_shapes[1]);
            default:
                LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
                return nullptr;
        }
    }

    void RuntimeOperatorUtils<float>::InitOperatorOutput(const std::vector<pnnx::Operator *> &pnnx_operators,
                                                         const std::vector<std::shared_ptr<RuntimeOperator> > &runtime_operators) {
        CHECK(!pnnx_operators.empty()) << "pnnx_operators is empty!!";
        CHECK(!runtime_operators.empty()) << "operators are empty!!";
        CHECK(pnnx_operators.size() == runtime_operators.size()) << "pnnx_operators size mismatch runtime_operators size!!";

        for (uint32_t i = 0; i <  pnnx_operators.size(); i++) {
            const std::vector<pnnx::Operand*>& pnnx_operands = pnnx_operators.at(i)->outputs;

            // get pnnx_operator with operands' shapes
            pnnx::Operand* pnnx_operand = pnnx_operands.front();
            std::vector<int32_t> pnnx_operand_shapes;
            std::copy_if(pnnx_operand->shape.begin(), pnnx_operand->shape.end(), std::back_inserter(pnnx_operand_shapes), [](const int32_t dim){return dim > 0;});

            // get runtime operator from runtime_operators vector
            // 这里本来想写成确切的类型的，但是考虑到以后想扩展支持计算节点的类型就用auto了，以后再改吧haha
            const auto& runtime_operator = runtime_operators.at(i);

            // get operand size
            const size_t pnnx_operand_size = std::accumulate(pnnx_operand_shapes.begin(), pnnx_operand_shapes.end(),  static_cast<size_t>(1), std::multiplies());

            const uint32_t batch = pnnx_operand_shapes.at(0);
            if (!runtime_operator->output_operand) {
                bool has_found = false;
                for (uint32_t j = 0; j <  i; j++) {
                    // 这里本来想写成确切的类型的，但是考虑到以后想扩展支持计算节点的类型就用auto了，以后再改吧haha
                    const auto& prev_runtime_operator = runtime_operators.at(j);
                    if (!prev_runtime_operator->output_operand || prev_runtime_operator->occur_end_time != -1) {
                        continue;
                    }

                    if (runtime_operator->start_time > prev_runtime_operator->occur_end_time) {
                        prev_runtime_operator->occur_end_time = -1;
                    }

                    if (runtime_operator->start_time > prev_runtime_operator->end_time) {
                        if (prev_runtime_operator->output_operand->size() == pnnx_operand_size) {
                            has_found = true;
                            // 这里本来想写成确切的类型的，但是考虑到以后想扩展支持计算节点的类型就用auto了，以后再改吧haha
                            const auto& prev_output_operand = prev_runtime_operator->output_operand;
                            runtime_operator->output_operand =
                                            std::make_shared<RuntimeOperand>(prev_output_operand->name + "_output", pnnx_operand_shapes, batch, RuntimeDataType::kTypeFloat32);

                            for (uint32_t b = 0; b < batch; b++) {
                                const ftensorptr prev_output_tensor = prev_output_operand->datas.at(b);
                                ftensorptr output_tensor = std::make_shared<ftensor>(prev_output_tensor->raw_ptr(), prev_output_tensor->shapes());

                                CheckAndReshapeTensor(output_tensor, pnnx_operand_shapes);
                                runtime_operator->output_operand->datas.at(b) = output_tensor;
                            }
                            prev_runtime_operator->occur_end_time = runtime_operator->end_time;
                        }
                    }
                }

                if (!has_found) {
                    std::vector<ftensorptr> output_operand_datas;
                    for (uint32_t j = 0; j < batch; j++) {
                        output_operand_datas.push_back(CreateTensor(pnnx_operand_shapes));
                    }
                    runtime_operator->output_operand = std::make_shared<RuntimeOperand>(
                                                        pnnx_operand->name + "_output", pnnx_operand_shapes, output_operand_datas, RuntimeDataType::kTypeFloat32);
                }
            } else {
                for (uint32_t b = 0; b < batch; b++) {
                    ftensorptr output_tensor = runtime_operator->output_operand->datas.at(b);
                    CheckAndReshapeTensor(output_tensor, pnnx_operand_shapes);
                }
            }
        }
    }
} // shukainfer