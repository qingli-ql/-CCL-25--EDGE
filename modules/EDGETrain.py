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

class EDGETrainer(Trainer):
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
        # gce
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, gce=True)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.accelerator.backward(loss, **kwargs)
        gce_grads = [param.grad.clone().detach() for name, param in model.named_parameters() if "lora" in name and param.grad is not None]
        self.optimizer.zero_grad()

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, gce=False)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.accelerator.backward(loss, **kwargs)
        erm_grads = [param for name, param in model.named_parameters() if "lora" in name and param.grad is not None]
        for grad, bias_grad in zip(erm_grads, gce_grads):
            erm_grad = grad.grad.clone()
            erm_grad = torch.nan_to_num(erm_grad.clone(), nan=0.0)
            bias_grad = torch.nan_to_num(bias_grad.clone(), nan=0.0)
            erm_grad_norm_value = torch.linalg.norm(erm_grad) or 1.0
            bias_grad_norm_value = torch.linalg.norm(bias_grad) or 1.0
            erm_grad_norm = erm_grad / erm_grad_norm_value
            bias_grad_norm = bias_grad / bias_grad_norm_value
            # common_grad 
            common_grad = erm_grad_norm + bias_grad_norm

            project_grad = []
            for i in range(erm_grad.size(0)):
                project_grad.append(torch.dot(bias_grad[i], common_grad[i]) * common_grad[i])
            project_grad = torch.stack(project_grad)
            ext_grad = bias_grad - project_grad
            # 并非自适应的减
            grad.grad = core_grad = erm_grad - self.alpha * ext_grad
            # grad.grad = common_grad * erm_grad_norm_value
            # 计算cosine信息， ratio信息
            # cosine = torch.dot(erm_grad_norm.flatten(), bias_grad_norm.flatten()).item()
            # if cosine>0:
            #         # 强假设, 充分而非必要条件 ！！！ 很难成立
            #     bias_projection = (
            #         torch.dot(erm_grad.flatten(), bias_grad_norm.flatten()) * bias_grad_norm 
            #     )
            #     core_grad = erm_grad - 1.0 * bias_projection
            #     grad.grad = core_grad

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def compute_loss(self, model, inputs, return_outputs=False, gce=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        inputs = {k:v for k,v in inputs.items() if k not in ['type']}
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        
        
        
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        self.label_smoother = CustomerLabelSmoother(epsilon=self.args.label_smoothing_factor)
        if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True, gce=gce)
        else:
            loss = self.label_smoother(outputs, labels, gce=gce)
        return (loss, outputs) if return_outputs else loss
    
@dataclass
class CustomerLabelSmoother:
    epsilon: float = 0.1
    ignore_index: int = -100
    def __call__(self, model_output, labels, shift_labels=False, gce=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        if gce:
            loss_fct = GeneralizedCELoss(q=0.7, ignore_index=self.ignore_index)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss
    
class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7, ignore_index=-100):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
        self.ignore_index = ignore_index
        self.NERO = 1e-6

    def forward(self, logits, targets, reduction="mean"):
        # print("logits shape:", logits.shape)
        # print("targets shape:", targets.shape)

        p = F.softmax(logits, dim=1)
        if torch.isnan(p).any() or torch.isinf(p).any():
            logits = torch.clamp(logits, min=-1e6, max=1e6)  # 限制 logits 范围
            p = F.softmax(logits, dim=1)

        NERO_ZERO = torch.tensor(self.NERO, device=p.device)
        NERO_ZERO = min(max(NERO_ZERO, 1e-6), 1-1e-6)  # 修正 self.NERO
        p = torch.nan_to_num(p, nan=NERO_ZERO, posinf=1-NERO_ZERO, neginf=NERO_ZERO)

        if np.isnan(p.mean().item()):
            raise NameError("GCE_p")

        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        epsilon = 1e-6
        Yg = torch.clamp(torch.nan_to_num(Yg, nan=0.0, posinf=1.0, neginf=0.0), min=epsilon, max=1.0)

        if np.isnan(Yg.mean().item()):
            raise NameError("GCE_Yg")

        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        loss = F.cross_entropy(logits, targets, reduction="none")
        mask = targets != self.ignore_index
        loss = loss * loss_weight
        loss = loss * mask.float()

        if reduction == "mean":
            mask_sum = mask.sum()
            if mask_sum == 0:
                return torch.tensor(0.0, device=logits.device)
            return loss.sum() / mask_sum
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss