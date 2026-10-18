/*
 * @file    runtime_graph.cpp
 * @author  nbunanana
 * @date    10/8/26
 * @brief   Runtime representation of a neural network graph
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */


#include "runtime/runtime_graph.hpp"
#include "code_status.hpp"
#include "runtime/pnnx/ir.h"
#include "runtime/runtime_attribute.hpp"
#include "runtime/runtime_datatype.hpp"
#include "runtime/runtime_operator.hpp"
#include "runtime/runtime_parameter.hpp"
#include <algorithm>
#include <cstdint>
#include <glog/logging.h>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace shukainfer {
    RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
        : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

    void RuntimeGraph::set_inputs(const std::string &input_name, const std::vector<ftensorptr> &inputs) {
        CHECK(this->graph_state_ == GraphState::Complete);
        std::shared_ptr<RuntimeOperator> input_operator;
        for (const auto& op : this->input_operators_) {
            if (op->name == input_name) {
                input_operator = op;
                break;
            }
        }
        CHECK(input_operator != nullptr) << "Can not find the input operator with name:" << input_name;
        PropagateLayerOutputs(input_operator, inputs);
    }

    std::vector<ftensorptr> RuntimeGraph::get_outputs(const std::string& output_name) const {
        CHECK(this->graph_state_ == GraphState::Complete);
        std::shared_ptr<RuntimeOperator> output_operator;
        for (const auto& op : this->output_operators_) {
            if(op->name == output_name) {
                output_operator = op;
                break;
            }
        }

        CHECK(output_operator != nullptr) << "Can not find the output operator with name:" << output_name;
        std::vector<ftensorptr> outputs;
        for (const auto& input_operand : output_operator->input_operands_seq) {
            std::copy(input_operand->datas.begin(), input_operand->datas.end(), std::back_inserter(outputs));
        }
        return outputs;
    }

    bool RuntimeGraph::is_input_op(const std::string &operator_name) const {
        for (auto& op: this->input_operators_) {
            CHECK(op != nullptr);
            if (op->name == operator_name) {
                return true;
            }
        }
        return false;
    }

    bool RuntimeGraph::is_output_op(const std::string &operator_name) const {
        for (auto& op : this->output_operators_) {
            CHECK(op != nullptr);
            if (op->name == operator_name) {
                return true;
            }
        }
        return false;
    }

    void RuntimeGraph::Build() {
        if (this->graph_state_ == GraphState::Complete) {
            LOG(INFO) << "Model has been built already!";
            return;       
        }

        if (this->graph_state_ == GraphState::NeedInit) {
            bool init_graph = Init();
            LOG_IF(FATAL, !init_graph || this->graph_state_ == GraphState::NeedInit) << "Init graph failed!";
        }

        CHECK(this->graph_state_ >= GraphState::NeedBuild) << "Graph status error, current state is " << int32_t(graph_state_);

        CreateNodeRelation();

        ReverseTopoSort();

        RuntimeOperatorUtils<float>::InitOperatorInput(this->operators_);
        RuntimeOperatorUtils<float>::InitOperatorOutput(this->graph_->ops, this->operators_);

        // 把pnnx的算子转换为shukainfer支持的算子后，graph_中存储的pnnx算子就会被释放掉.
        this->graph_state_ = GraphState::Complete;
        if(this->graph_ != nullptr) {
            this->graph_.reset(); // 先delete对象，再给自己置nullptr.
        }
    }

    void RuntimeGraph::set_bin_path(const std::string& bin_path) {
        this->bin_path_ = bin_path;
    }

    void RuntimeGraph::set_param_path(const std::string& param_path) {
        this->param_path_ = param_path;
    }

    const std::string& RuntimeGraph::get_param_path() const {
        return this->param_path_;
    }

    const std::string &RuntimeGraph::get_bin_path() const {
        return this->bin_path_;
    }

    template<typename T>
    StatusCode ExecuteLayer(const std::shared_ptr<Layer<T>>& layer, const std::string& operator_name, const std::string& operator_type, bool is_debug) {
        CHECK(layer != nullptr);
        StatusCode status;
        if(is_debug) {
            utils::LayerTimeLogging layer_time_logging(operator_name, operator_type);
            status = layer->Forward();
        } else {
            status = layer->Forward();
        }
        return status;
    }

    void RuntimeGraph::Forward(bool debug) {
        if (this->graph_state_ < GraphState::Complete) {
            LOG(FATAL) << "Graph need be build!"
                    << ", current state is " << int32_t(graph_state_);
        }

        if (debug) {
            utils::LayerTimeStatesSingleton::LayerTimeStatesCollectorInit();
        }

        for (const auto& current_op : this->operators_) {
            current_op->has_forward = false;
            CHECK_GT(current_op->start_time, 0);

            if (is_input_op(current_op->name) || is_output_op(current_op->name)) {
                current_op->has_forward = true;
                continue;
            }

            CHECK(current_op->layer != nullptr)
                << "The layer corresponding to the op " << current_op->name
                << " is empty, indicating that it may not have been created.";

            StatusCode status = ExecuteLayer(current_op->layer, current_op->name, current_op->type, debug);
            CHECK(status == StatusCode::kSuccess)
                << current_op->layer->layer_name()
                << " layer forward failed, error code: " << static_cast<int32_t>(status);

            current_op->has_forward = true;
            // 将当前节点的output_operand广播到下游节点的input_operands
            PropagateLayerOutputs(current_op, current_op->output_operand->datas);
        }

        if (debug) {
            utils::LayerTimeLogging::SummaryLogging();
        }

        for (const auto& op : this->operators_) {
            LOG_IF(FATAL, !op->has_forward) << "The operator: " << op->name << " has not been forward yet!";
        }
    }

    // Only use in this cpp file
    static bool IsQuantizeOperator_(const pnnx::Operator* op) {
        if(op != nullptr) {
            return false;
        }
        return op->name.rfind("quantized_", 0) == 0; 
    }

    bool RuntimeGraph::Init() {
        if (this->bin_path_.empty()) {
            LOG(ERROR) << "The bin path is empty!!";
            return false;  
        }
        if (this->param_path_.empty()) {
            LOG(ERROR) << "The param path is empty!!";
            return false;
        }

        this->graph_ = std::make_unique<pnnx::Graph>();
        int32_t load_graph_result = this->graph_->load(this->param_path_, this->bin_path_);
        if(load_graph_result != 0) {
            LOG(ERROR) << "Can not find the param path or bin path:" << this->param_path_ << " " << this->bin_path_;
            return false;
        }

        std::vector<pnnx::Operator*> operators = this->graph_->ops;
        if (operators.empty()) {
            LOG(ERROR) << "Can not read the layers' define";
            return false;
        }

        this->operators_.clear();
        for(const pnnx::Operator* op : operators) {
            if(op == nullptr) {
                LOG(ERROR) << "Meet the empty node in the model!";
                continue;
            } else {
                if(!IsQuantizeOperator_(op)) {
                    std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();

                    // Initialize operator name and type.
                    runtime_operator->name = op->name;
                    runtime_operator->type = op->type;

                    // Initialize operator input.
                    InitGraphOperatorsInput(op->inputs, runtime_operator);

                    // Record output name with operand.
                    InitGraphOperatorsOutput(op->outputs, runtime_operator);

                    // Initialize attribute(weights) in operator.
                    InitGraphAttributes(op->attrs, runtime_operator);

                    // Initialize parameter in operator.
                    InitGraphParameters(op->params, runtime_operator);

                    this->operators_.push_back(runtime_operator);
                } else {
                    LOG(FATAL) << "UnSupported quantize operator in the model " << op->name
                   << " type: " << op->type;
                }
            }
        }

        this->graph_state_ = GraphState::NeedBuild;
        return true;
    }

    void RuntimeGraph::ReverseTopoSort() {
        for (const auto& op : this->operators_) {
           if (op != nullptr && !op->has_forward) {
                // 根据输入节点构建逆拓扑排序
                int32_t current_forward_idx = 0;
                this->ReverseTopoSortInternal(op, current_forward_idx);
           }
        }

        std::sort(this->operators_.begin(), this->operators_.end(), [](const auto& op1, const auto& op2){
            return op1->start_time > op2->start_time;
        });

        // 给operator的start_time赋值
        int32_t forward_index = 1;
        for(const auto& op : this->operators_) {
            op->start_time = forward_index;
            forward_index +=1;
        }

        // 给operator的end_time赋值
        for(const auto& op : this->operators_) {
            const auto& next_operators = op->output_operators;
            int32_t last_forward_index = -1;
            for(const auto& [_, next_op] : next_operators) {
                if(next_op->start_time >= last_forward_index) {
                    last_forward_index = next_op->start_time;
                }
            }

            if(last_forward_index == -1) {
                op->end_time = op->start_time + 1;
            } else {
                op->end_time = last_forward_index;
            }
            op->occur_end_time = -1;
        }
    }

    template<typename T>
    void RuntimeGraph::ReverseTopoSortInternal(const std::shared_ptr<RuntimeOperatorBase<T> > &root_operator, int32_t &current_forward_idx) {
        if(!root_operator) {
            LOG(INFO) << "Current operator is nullptr";
            return;
        }

        if(root_operator->input_operands.empty() && ! root_operator->has_forward) {
            this->input_operators_.push_back(root_operator);
        }
        if(root_operator->output_names.empty() && !root_operator->has_forward) {
            this->output_operators_.push_back(root_operator);
        }

        root_operator->has_forward = true;
        const auto& next_operators = root_operator->output_operators;
        for(const auto& [_, op] : next_operators) {
            if(op != nullptr && !op->has_forward) {
                this->ReverseTopoSortInternal(op, current_forward_idx);
            }
        }

        for (const auto& [_, op] : next_operators) {
            CHECK_EQ(op->has_forward, true);
        }
        root_operator->start_time = current_forward_idx;
        current_forward_idx += 1;
    }

    void RuntimeGraph::CreateNodeRelation() {
        for (const auto& current_operator : this->operators_) {
            for (const auto& KOutputName : current_operator->output_names) {
                for (const auto& output_operator : this->operators_) {
                    if (output_operator != current_operator && output_operator->name == KOutputName) {
                        current_operator->output_operators.insert({KOutputName, output_operator});
                    }
                }
            }
            if (current_operator->type != "pnnx.Input" && current_operator->type != "pnnx.Output") {
                auto layer = RuntimeGraph::CreateLayer(current_operator);
                if(layer != nullptr) {
                    current_operator->layer = layer;
                    layer->set_runtime_operator(current_operator);
                } else {
                    LOG(FATAL) << "Layer: " << current_operator->name << "Create failed!!"; 
                }
            }
        }
    }

    template<typename T>
    void RuntimeGraph::InitGraphOperatorsInput(const std::vector<pnnx::Operand *> inputs, const std::shared_ptr<RuntimeOperatorBase<T> > &runtime_operator) {
        if(inputs.empty()) {
            return;
        }
        CHECK(runtime_operator != nullptr) << "The runtime is null pointer";

        for(const pnnx::Operand* input : inputs) {
            if(input == nullptr) {
                continue;
            }

            std::vector<int32_t> dims;
            std::copy(input->shape.begin(), input->shape.end(), std::back_inserter(dims));
            
            CHECK(!dims.empty());
            std::shared_ptr<RuntimeOperandBase<T>> runtime_operand = std::make_shared<RuntimeOperandBase<T>>();
            runtime_operand->name = input->producer->name;
            runtime_operand->shapes = dims;
            runtime_operator->input_operands.insert({input->producer->name, runtime_operand});
            runtime_operator->input_operands_seq.push_back(runtime_operand);

            switch (input->type) {
                case 1: {
                    runtime_operand->type = RuntimeDataType::kTypeFloat32;
                    break;
                }
                case 2: {
                    runtime_operand->type = RuntimeDataType::kTypeInt8;
                    break;
                }
                default: {
                    LOG(FATAL) << "Unknow input operand type:" << input->type;
                }
            }
        }
    }

    template<typename T>
    void RuntimeGraph::InitGraphOperatorsOutput(const std::vector<pnnx::Operand *> outputs, const std::shared_ptr<RuntimeOperatorBase<T> > &runtime_operator) {
        if (outputs.empty()) {
            return;
        }

        CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer!!";
        for (const pnnx::Operand* output : outputs) {
            if (output == nullptr) {
                continue;
            }
            
            for (const pnnx::Operator* con : output->consumers) {
                runtime_operator->output_names.push_back(con->name);
            }
        }
    }

    template<typename T>
    void RuntimeGraph::InitGraphAttributes(const std::map<std::string, pnnx::Attribute> &attributes, const std::shared_ptr<RuntimeOperatorBase<T> > &runtime_operator) {
        if (attributes.empty()) {
            return;
        }

        CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer!!";
        for(const auto& [name, attr] : attributes) {
            switch (attr.type) {
                case 1: {
                    std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>(
                        attr.shape, RuntimeDataType::kTypeFloat32, attr.data); 
                        runtime_operator->attributes.insert(name, runtime_operator);
                        break;
                }
                default: {
                    LOG(FATAL) << "Unknow attribute type:" << attr.type;
                    break;
                }
            }
        }

    }

    template<typename T>
    void RuntimeGraph::InitGraphParameters(const std::map<std::string, pnnx::Parameter> &parameters, const std::shared_ptr<RuntimeOperatorBase<T> > &runtime_operator) {
        if (parameters.empty()) {
            return;
        }
        CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
        for (const auto& [name, param] : parameters) {
            switch (param.type) {
                case int32_t(RuntimeParameterType::kParameterUnknown): {
                    std::shared_ptr<RuntimeParameter> runtime_parmeter = 
                        std::make_shared<RuntimeParameter>();
                    runtime_operator->parameters.insert(name, runtime_parmeter);
                    break;
                }

                case int32_t(RuntimeParameterType::kParameterInt): {
                    std::shared_ptr<RuntimeParameterInt> runtime_parameter = 
                        std::make_shared<RuntimeParameterInt>(param.i);
                    runtime_operator->parmeters.insert(name, runtime_parameter);
                    break;
                }

                case int32_t(RuntimeParameterType::kParameterFloat): {
                    std::shared_ptr<RuntimeParameterFloat> runtime_parameter = 
                        std::make_shared<RuntimeParameterFloat>(param.f);
                    runtime_operator->parmeters.insert(name, runtime_parameter);
                    break;
                }

                case int32_t(RuntimeParameterType::kParameterString): {
                    std::shared_ptr<RuntimeParameterString> runtime_parameter = 
                        std::make_shared<RuntimeParameterString>(param.s);
                    runtime_operator->parmeters.insert(name, runtime_parameter);
                    break;
                }

                case int32_t(RuntimeParameterType::kParameterBool): {
                    std::shared_ptr<RuntimeParameterBool> runtime_parameter = 
                        std::make_shared<RuntimeParameterBool>(param.b);
                    runtime_operator->parmeters.insert(name, runtime_parameter);
                    break;
                }

                case int32_t(RuntimeParameterType::kParameterIntArray): {
                    std::shared_ptr<RuntimeParameterIntArray> runtime_parameter = 
                        std::make_shared<RuntimeParameterIntArray>(param.ai);
                    runtime_operator->parmeters.insert(name, runtime_parameter);
                    break;
                }

                case int32_t(RuntimeParameterType::kParameterStringArray): {
                    std::shared_ptr<RuntimeParameterStringArray> runtime_parameter = 
                        std::make_shared<RuntimeParameterStringArray>(param.as);
                    runtime_operator->parmeters.insert(name, runtime_parameter);
                    break;
                }

                case int32_t(RuntimeParameterType::kParameterFloatArray): {
                    std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter = 
                        std::make_shared<RuntimeParameterFloatArray>(param.af);
                    runtime_operator->parmeters.insert(name, runtime_parameter);
                    break;
                }
            }
        }
    }

    template<typename T>
    std::shared_ptr<Layer<T>> RuntimeGraph::CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>>& common_operator) {
        LOG_IF(FATAL, !common_operator) << "Operator is empty!";
        auto layer = LayerRegisterer::CreateLayer(common_operator);
        LOG_IF(FATAL, !layer) << "Layer init failed " << common_operator->type;
        return layer;
    }

    template<typename T>
    void RuntimeGraph::PropagateLayerOutputs(const std::shared_ptr<RuntimeOperatorBase<T> > &current_operator, const std::vector<std::shared_ptr<Tensor<T> > > &layer_output_datas) {
        for (const auto& [_, output_operator] : current_operator->output_operators) {
            const auto& next_input_operands = output_operator->input_operands;
            const auto& next_input_operand_iter = next_input_operands.find(current_operator->name);
            if (next_input_operand_iter != next_input_operands.end()) {
                std::vector<tensorptr<T>>& next_input_operand_datas = next_input_operand_iter->second->datas;

                for (uint32_t i = 0; i < next_input_operand_datas.size(); i++) {
                    const tensorptr<T>& layer_output_data = layer_output_datas.at(i);
                    if (next_input_operand_datas.at(i) != nullptr) {
                        CHECK(next_input_operand_datas.at(i)->shapes() == layer_output_data->shapes());
                    }
                    next_input_operand_datas.at(i) = layer_output_data;
                }
            }
        }
    }

    RuntimeGraph::GraphState RuntimeGraph::get_graph_state() const {
        return this->graph_state_;
    }
}// shukainfer