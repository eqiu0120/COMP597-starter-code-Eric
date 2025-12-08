"""Various models to train.

This module provides various objects that are designed to train machine
learning models. Please refer to each repesctive classes to know how to use
them.

This module provides a factory that can be used to construct a variety of 
models to train using the trainers provided by the trainer module. Please each 
model's directory for model specific documentation.

"""
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple
import importlib
import pkgutil
import src.config as config
import src.trainer as trainer
import torch.utils.data

_MODEL_INIT_FUNCTION_NAME = "init_model"
_MODEL_NAME_VARIABLE_NAME = "model_name"
_MODELS = {}

def model_init_fallback(*args, **kwargs):
    raise Exception("Failed to find the init model function")

def discover_submodules() -> List[pkgutil.ModuleInfo]:
    submodules = []
    for submodule in pkgutil.iter_modules(path=__path__):
        submodules.append(submodule)
    return submodules

def import_submodule_if_contains_model(submodule : pkgutil.ModuleInfo, model_init_function_name : str) -> Optional[ModuleType]:
    if not submodule.ispkg:
        return None
    try:
        module = importlib.import_module(name=f".{submodule.name}", package=__package__)
        if getattr(module, model_init_function_name, None) is None:
            return None
    except Exception:
        print(f"Failed to import {submodule.name}")
        return None
    return module

def register_model(models : Dict[str, Callable], module : ModuleType, model_name_attribute_name : str, model_init_function_name : str) -> Dict[str, Callable]:
    default_name = module.__package__.split(".")[-1]
    name = getattr(module, model_name_attribute_name, default_name)
    models[name] = getattr(module, model_init_function_name, model_init_fallback)
    return models

def register_models() -> Dict[str, Callable]:
    submodules = discover_submodules()
    models = {}
    for submodule in submodules:
        module = import_submodule_if_contains_model(submodule, _MODEL_INIT_FUNCTION_NAME)
        if module is None:
            # print(f"Found submodule {submodule.name} but it did not fulfill requirements to register a model")
            continue
        models = register_model(models, module, _MODEL_NAME_VARIABLE_NAME, _MODEL_INIT_FUNCTION_NAME)
    return models

_MODELS = register_models()

def model_factory(conf : config.Config, dataset : torch.utils.data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    global _MODELS
    init_func = _MODELS.get(conf.model, None)
    if init_func is None:
        raise Exception(f"Unknown model {conf.model}")
    return init_func(conf, dataset)

def get_available_models() -> List[str]:
    global _MODELS
    return [m for m in _MODELS.keys()]
