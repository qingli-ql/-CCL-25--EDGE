import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)


class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets, reduction="mean"):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError("GCE_p")
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError("GCE_Yg")

        loss = F.cross_entropy(logits, targets, reduction="none") * loss_weight

        return loss


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.A = 1e-7

    def forward(self, pred, labels, index=None, mode="sce"):
        # CCE
        ce = F.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, pred.shape[-1]).float()
        label_one_hot = torch.clamp(label_one_hot, min=self.A, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        if mode == "ce":
            loss = ce
        else:
            loss = self.alpha * ce + self.beta * rce
        return loss


class GSCE(nn.Module):
    def __init__(self, alpha=0.1, beta=1, q=0.7):
        super(GSCE, self).__init__()
        self.gce = GeneralizedCELoss(q=q)
        self.sce = SymmetricCrossEntropy(alpha=alpha, beta=beta)

    def forward(self, logits, labels):
        # KL(p|q) + KL(q|p)
        return 0.1 * self.gce(logits, labels) + self.sce(logits, labels)


class BiasedFocalLoss(nn.Module):
    def __init__(self, q=0.7):
        super(BiasedFocalLoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)

        loss = F.cross_entropy(logits, targets, reduction="none")

        return loss
