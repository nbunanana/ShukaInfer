/*
 * @file    layer_factory.cpp
 * @author  nbunanana
 * @date    10/14/26
 * @brief   The registry implementation of the layer, including the operator registration class and the operator registry recycling class.
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */

#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_graph.hpp"


namespace shukainfer {
    // Global registry
    template <typename T>
    typename LayerRegister<T>::CreateRegistry* LayerRegister<T>::registry_ = nullptr;

    template <typename T>
    void LayerRegister<T>::RegisterCreator(const std::string& layer_type, const Creator& creator) {
        CHECK(!layer_type.empty()) << "Layer type is empty!";
        CHECK(creator != nullptr) << "Creator is null!";

        CreateRegistry* registry = registry_;

        // Check the table to see if this operator has been registered
        CHECK_EQ(registry->count(layer_type), 0) << "Layer type: " << layer_type << "has already registered!";
        registry->insert({layer_type, creator});
    }

    template <typename T>
    std::shared_ptr<Layer<T>> LayerRegister<T>::CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>>& op) {
        CreateRegistry* registry = Registry();
        const std::string& layer_type = op->type;
        LOG_IF(FATAL, registry->count(layer_type) == 0) << "Can not find the layer type: " << layer_type;
        const auto& creator = registry->find(layer_type)->second;

        LOG_IF(FATAL, creator != nullptr) << "Layer creator is null!";
        std::shared_ptr<Layer<T>> layer;

        const auto& status = creator(op, layer);
        LOG_IF(FATAL, status != StatusCode::kSuccess) << "Create the layer: " << layer_type << " failed, error code: " << static_cast<int32_t>(status);
        return layer;
    }

    template <typename T>
    typename LayerRegister<T>::CreateRegistry* LayerRegister<T>::Registry() {
        if (registry_ == nullptr) {
            registry_ = new CreateRegistry();
            static RegistryGarbageCollector<T> c;
        }
        CHECK(registry_ != nullptr) << "Global layer register init failed!";
        return registry_;
    }

    template<typename T>
    std::vector<std::string> LayerRegister<T>::layer_types() {
        std::set<std::string> layer_types_unique;
        CreateRegistry* registry = Registry();
        for (const auto& [layer_type, _] : *registry) {
            layer_types_unique.insert(layer_type);
        }
        std::vector<std::string> layer_types(layer_types_unique.begin(), layer_types_unique.end());
        return layer_types;
    }

    template class LayerRegister<float>;
    template class LayerRegister<uint8_t>;
} // shukainfer
