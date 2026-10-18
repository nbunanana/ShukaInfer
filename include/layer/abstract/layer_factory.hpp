/*
 * @file    layer_factory.hpp
 * @author  nbunanana
 * @date    10/13/26
 * @brief   
 * @copyright Copyright (c) 2026 shuka
 * @license MIT
 */
#ifndef SHUKAINFER_LAYER_ABSTRACT_LAYERFACTORY_HPP
#define SHUKAINFER_LAYER_ABSTRACT_LAYERFACTORY_HPP

#include <map>
#include <string>
#include "layer.hpp"
#include "runtime/runtime_operator.hpp"

namespace shukainfer {
    template <typename T>
    class LayerRegister {
    private:
        typedef StatusCode (*Creator)(const std::shared_ptr<RuntimeOperatorBase<T>>& op, std::shared_ptr<Layer<T>>& layer);

        typedef std::unordered_map<std::string, Creator> CreateRegistry;

    public:
        template <typename U>
        friend class LayerRegisterWarper;

        template <typename U>
        friend class RegistryGarbageCollector;

        static void RegisterCreator(const std::string& layer_type, const Creator& creator);

        static std::shared_ptr<Layer<T>> CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>>& op);

        static CreateRegistry* Registry();

        static std::vector<std::string> layer_types();

    private:
        static CreateRegistry* registry_;
    };


    template <typename T>
    class LayerRegisterWarper {
    public:
        explicit LayerRegisterWarper(const typename LayerRegister<T>::Creator& creator, const std::string& layer_type) {
            LayerRegister<T>::RegisterCreator(layer_type, creator);
        }

        template <typename... Ts>
        explicit LayerRegisterWarper(const typename LayerRegister<T>::Creator& creator, const std::string& layer_type, const Ts&... other_layer_types)
            : LayerRegisterWarper(creator, other_layer_types...)
        {
            LayerRegister<T>::RegisterCreator(layer_type, creator);
        }

        explicit LayerRegisterWarper(const typename LayerRegister<T>::Creator& creator){}
    };

    template <typename T>
    class RegistryGarbageCollector {
    public:
        ~RegistryGarbageCollector() {
            if (LayerRegister<T>::registry_ != nullptr) {
                delete LayerRegister<T>::registry_;
                LayerRegister<T>::registry_ = nullptr;
            }
        }

        friend class LayerRegister<T>;

    private:
        RegistryGarbageCollector() = default;
        RegistryGarbageCollector(const RegistryGarbageCollector&) = default;
    };
} // shukainfer
#endif //SHUKAINFER_LAYER_ABSTRACT_LAYERFACTORY_HPP
