from transformers.trainer import (
    Trainer,
    _is_peft_model,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)   
from transformers.utils import (
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_in_notebook,
    is_ipex_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    strtobool,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass, field
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class BaseTrainer(Trainer):
    def setMode(self, mode, alpha=1.0):
        self.mode = mode
        self.alpha = alpha

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        # 计算loss 以及 backward
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        # 计算loss 以及 backward
        self.optimizer.zero_grad()
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.accelerator.backward(loss, **kwargs)
        # erm_grads = [param for name, param in model.named_parameters() if "lora" in name and param.grad is not None]
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        inputs = {k:v for k,v in inputs.items() if k not in ['type']}
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # print(">>> logits.shape:", logits.shape)
        # print(">>> labels.shape:", labels.shape)
        # print(">>> labels.min():", labels.min().item())
        # print(">>> labels.max():", labels.max().item())
        
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        self.label_smoother = CustomerLabelSmoother(epsilon=self.args.label_smoothing_factor)
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
        return (loss, outputs) if return_outputs else loss
    
@dataclass
class CustomerLabelSmoother:
    epsilon: float = 0.1
    ignore_index: int = -100
    def __call__(self, model_output, labels, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss
    