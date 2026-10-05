/*
 * @file    runtime_operand.hpp
 * @author  nbunanana
 * @date    9/27/26
 * @brief  Base of runtime graph operand.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_RUNTIME_RUNTIMEOPERAND_HPP
#define SHUKAINFER_RUNTIME_RUNTIMEOPERAND_HPP
#include <memory>
#include <vector>
#include <string>
#include "data/tensor.hpp"
#include "runtime_datatype.hpp"

namespace shukainfer {
    /**
     * @brief Base of runtime graph operand.
     *
     * @tparam T Operand data type(float、int、etc.).
     */
    template<typename T>
    struct RuntimeOperandBase {
    public:
        explicit  RuntimeOperandBase() = default;

        /**
         *
         * @param name Name of the Operand.
         * @param shapes Shapes of the operand.
         * @param datas Vector containing operand data.
         * @param type Data type the operand.
         */
        explicit  RuntimeOperandBase(std::string name, std::vector<int32_t> shapes, std::vector<std::shared_ptr<Tensor<T>>> datas,  RuntimeDataType type)
                                : name(std::move(name)),  shapes(std::move(shapes)), datas(std::move(datas)), type(type) {}

        /**
         *
         * @param name Name of the Operand.
         * @param shapes Shapes of the operand.
         * @param data_size Vector containing operand size with data.
         * @param type Data type the operand.
         */
        explicit RuntimeOperandBase(std::string name, std::vector<int32_t> shapes, uint32_t data_size, RuntimeDataType type)
        : name(std::move(name)), shapes(std::move(shapes)), type(type) {
            datas.resize(shapes.size());
        }

        /**
         * @brief Get size with shapes
         * @return size with shapes.
         */
        size_t size() const;

        /**
         * @brief Name of the Operand.
         */
        std::string name;

        /**
         * @brief shapes of the operand.
         */
        std::vector<int32_t> shapes;

        /**
         * @brief Vector containing operand data。
         */
        std::vector<std::shared_ptr<Tensor<T>>> datas;

        /**
         * @brief Data type the operand.
         */
        RuntimeDataType type = RuntimeDataType::KTypeUnknow;
    };

    using RuntimeOperand = RuntimeOperandBase<float>;
    using RuntimeOperandQuantized = RuntimeOperandBase<int8_t>;
}  // shuka_infer
#endif //SHUKAINFER_RUNTIME_RUNTIMEOPERAND_HPP
