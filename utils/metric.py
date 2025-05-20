import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    accuracy_score,
    f1_score,
    label_ranking_loss,
    coverage_error,
)
import torch
import torch.nn.functional as F

my_binary_criterion = F.cross_entropy
NERO_ZERO = 1e-10


def get_grad(targets, logits):
    # gard following GA
    loss = my_binary_criterion(logits, targets, reduction="item")
    _grad = 1.0 - torch.exp(-loss).detach().clone()
    return _grad.sum().item()


def get_entropy(targets, logits, reduction="mean"):
    # following DENEB
    logits = torch.clamp(logits, NERO_ZERO, 1)
    _logits = torch.clamp(1 - logits, NERO_ZERO, 1)
    entropy = -(logits * torch.log(logits) + _logits * torch.log(_logits))
    if reduction == "item":
        return entropy
    return entropy.mean().item()


def get_metrics(metric):
    if metric == "auc":
        return roc_auc_score
    elif metric == "entropy":
        return get_entropy
    elif metric == "grad":
        return get_grad
    elif metric == "acc":
        return lambda y_gt, y_pd: (y_gt == y_pd).float().mean().item() * 100
    elif metric == "loss":
        return F.cross_entropy
    else:
        raise NotImplementedError
