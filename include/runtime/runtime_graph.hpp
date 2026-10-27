/*
 * @file    runtime_graph.hpp
 * @author  nbunanana
 * @date    10/7/26
 * @brief   Runtime representation of a neural network graph
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 *
 */
#ifndef SHUKAINFER_RUNTIME_RUNTIMEGRAPH_HPP
#define SHUKAINFER_RUNTIME_RUNTIMEGRAPH_HPP

#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include "runtime/runtime_operand.hpp"
#include "runtime/runtime_operator.hpp"
#include "layer/abstract/layer.hpp"
#include "runtime/pnnx/ir.h"

namespace shukainfer {
    /**
     * @brief Runtime representation of a neural network graph
     *
     * This class builds a runtime representation of a neural network
     * computation graph from saved model files. It initializes the graph
     * topology and parameters, sets graph inputs, performs graph execution,
     * and retrieves outputs.
     */
    class RuntimeGraph {
    public:
        /**
         * @brief   Construct a new RuntimeGraph object.
         *
         * Creates a runtime graph representation from parameter and bin file.
         *
         * @param param_path Path to the parameter file defining the graph structure.
         * @param bin_path Path to the bin file containing the graph weights.
         */
        RuntimeGraph(std::string param_path, std::string bin_path);

        /**
         * @brief Sets the inputs to the graph.
         *
         * Sets the input tensors for executing the graph.
         *
         * @param input_name Name of the input.
         * @param inputs Vector of input tensors.
         */
        void set_inputs(const std::string& input_name, const std::vector<ftensorptr>& inputs);

        /**
         * @brief Gets output tensors from the graph.
         *
         * Returns the output tensors with the given name.
         *
         * @param output_name Name of the graph output.
         * @return Vector of output tensors.
         */
        std::vector<ftensorptr> get_outputs(const std::string& output_name) const;

        /**
         * @brief Checks if an operators is an input operators.
         *
         * Returns true if the given op name corresponds to an input operation.
         *
         * @param operator_name Name of the operator to check
         * @return True if the operator is an input operator,false otherwise.
         */
        bool is_input_op(const std::string& operator_name) const;

        /**
         * @brief Checks if an operator is an output operator.
         * @param operator_name Checks if an operator is an output operator.
         * @return True if the operator is an input operator,
         *          false otherwise.
         */
        bool is_output_op(const std::string& operator_name) const;

        /**
         * @brief Builds the runtime graph.
         *
         * Performs the actual construction of the runtime graph based on
         * initialized operator and parameters.
         */
        void Build();

        /**
         * @brief Sets the path to the weights file.
         * @param bin_path Path to the weights file.
         */
        void set_bin_path(const std::string& bin_path);

        /**
         * @brief Sets the path to the parameter file.
         * @param param_path Path to the parameter file.
         */
        void set_param_path(const std::string& param_path);

        /**
         * @brief Gets the path to the weights file.
         * @return Path to the weights file.
         */
        const std::string& get_bin_path() const;

        /**
         * @brief Gets the path to the parameter file.
         * @return Path to the parameter file.
         */
        const std::string& get_param_path() const;

        /**
         * @brief Executes the computation graph.
         *
         * Executes the graph operations in depth-first order.
         *
         * @param debug Whether to print debugging information during execution.
         */
        void Forward(bool debug = false);

    private:
        /**
         * @brief Initializes the graph.
         *
         * Initializes the graph from the provided parameter and weights files.
         *
         * @return True if initialization succeeded, false otherwise.
         */
        bool Init();

        /**
         * @brief performs reverse topological sort on the graph.
         *
         * Sorts the graph operators in reverse topological order for execution.
         */
        void ReverseTopoSort();

        /**
         * @brief Recursive reserve topological sort.
         * @tparam T The operator type.
         * @param root_operator Root operator to start sort form !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         * @param current_forward_idx
         */
        template<typename T>
        void ReverseTopoSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>>& root_operator, int32_t& current_forward_idx);

        /**
         * @brief Creates graph node relations
         *
         * Connects graph operators based on their input and output relation to construct the graph topology.
         */
        void CreateNodeRelation();

        /**
         * @brief Initializes operator inputs.
         *
         * Initializes the input tensors of a graph operator.
         *
         * @tparam T The operator type.
         * @param inputs Operands inputs in PNNX graph.
         * @param runtime_operator The runtime operator to initialize inputs for.
         */
        template<typename T>
        static void InitGraphOperatorsInput(const std::vector<pnnx::Operand*> inputs, const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);

        /**
         * @brief Initializes operator outputs.
         *
         *  Initializes the output tensors of a graph operator.
         *
         * @tparam T The operator typ.
         * @param outputs Operands outputs in PNNX graph.
         * @param runtime_operator The runtime operator to initialize inputs for.
         */
        template<typename T>
        static void InitGraphOperatorsOutput(const std::vector<pnnx::Operand*>& outputs, const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);

        /**
         * @brief Initializes operator attributes.
         *
         * Initializes the attributes of a graph operator.
         *
         * @tparam T The operator type.
         * @param attributes Operator attributes in PNNX graph.
         * @param runtime_operator The runtime operator to initialize attributes for.
         */
        template<typename T>
        static void InitGraphAttributes(const std::map<std::string, pnnx::Attribute>& attributes, const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);

        /**
         * @brief  Initializes operator parameters.
         *
         * Initializes the parameters of a graph operator.
         *
         * @tparam T The operator type.
         * @param parameters Operator parameters in PNNX graph.
         * @param runtime_operator The runtime operator to initialize parameters for.
         */
        template<typename T>
        static void InitGraphParameters(const std::map<std::string, pnnx::Parameter>& parameters, const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);

        /**
         * @brief Creates a Layer object for a graph operator.
         *
         * Creates and initializes a Layer object corresponding to the given RuntimeOperator.
         *
         * @tparam T The operator type.
         * @param common_operator The RuntimeOperator to create a Layer for.
         * @return Pointer to the created Layer object.
         */
        template<typename T>
        static std::shared_ptr<Layer<T>> CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>>& common_operator);

        /**
         * @brief Propagate output data.
         *
         * Propagate output data from current operator to inputs of next operators.
         *
         * @tparam T The operator type.
         * @param current_operator Current operator whose outputs will be propagated.
         * @param layer_output_datas Output data from current operator.
         */
        template<typename T>
        static void  PropagateLayerOutputs(const std::shared_ptr<RuntimeOperatorBase<T>>& current_operator, const std::vector<std::shared_ptr<Tensor<T>>>& layer_output_datas);

    private:
        enum class GraphState {
            NeedInit = -2,
            NeedBuild = -1,
            Complete = 0,
        };

    public:
        /**
         * @brief Gets current graph state.
         * @return Current graph state.
         */
        GraphState get_graph_state() const;

    private:
        std::string param_path_;

        std::string bin_path_;

        std::unique_ptr<pnnx::Graph> graph_;

        GraphState graph_state_ = GraphState::NeedInit;

        std::vector<std::shared_ptr<RuntimeOperator>> input_operators_;

        std::vector<std::shared_ptr<RuntimeOperator>> output_operators_;

        std::vector<std::shared_ptr<RuntimeOperator>> operators_;
    };
} // shukainfer
#endif //SHUKAINFER_RUNTIME_RUNTIMEGRAPH_HPP
