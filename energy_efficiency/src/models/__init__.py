"""Various models to train.

This module provides various objects that are designed to train machine
learning models. Please refer to each repesctive classes to know how to use
them.

This module provides a factory that can be used to construct a variety of 
models to train using the trainers provided by the trainer module. Please each 
model's directory for model specific documentation.

"""
from src.models import (
    gpt2 as gpt2,)
from typing import Any, Dict, Optional, Tuple
import src.config as config
import src.trainer as trainer
import torch.utils.data

def model_factory(conf : config.Config, dataset : torch.utils.data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    if conf.model == "gpt2":
        return gpt2.gpt2_init(conf, dataset)
    else:
        raise Exception(f"Unknown model {conf.model}")
