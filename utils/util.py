import io
import torch
import numpy as np
import json
import pandas as pd
import logging
import time
import os


class MultiDimAverageMeter(object):
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(*dims)

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        # 计算acc，正确的次数
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        # 计算当前位置所统计的次数
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )

    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()


class EMA:

    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))

    def update(self, data, index):
        self.parameter[index] = (
            self.alpha * self.parameter[index]
            + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()


class MyEMA:
    def __init__(self, size, alpha=0):
        self.alpha = alpha
        self.parameter = torch.zeros(size)
        self.updated = torch.zeros(size)

    def update(self, data, index):
        self.parameter[index] = (
            self.alpha * self.parameter[index]
            + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

    def get(self, index):
        return self.parameter[index]


def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def detect_anomaly(loss):
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Loss is infinite or NaN!")


def nli_task_dict(base_path="F:\\Datasets\\NLI\\test\\"):
    task_dict = {
        "cola": {},
        "mnli": {
            "path": base_path + "MNLI_hard_m\\test.jsonl",
            "keys": ("premise", "hypothesis", "gold"),
        },
        "mrpc": {},
        "qnli": {},
        "qqp": {},
        "rte": {},
        "wnli": {},
        "snli": {},
        "snli_disco": {},
        "PI_CD": {
            "path": base_path + "PI_CD\\snli\\snli_1.0_test_hard.jsonl",
            "keys": ("sentence1", "sentence2", "gold_label"),
        },  ## SNLI-HARD test  3261
        "PI_SP": {
            "path": base_path + "PI_SP\\mnli\\mismatch_dev_hard_lambda=0.7.tsv",
            "keys": ("sentence1", "sentence2", "gold_label"),
        },  ## mnli-hard mismatch_lambda07 371
        "IC_CS": {
            "path": base_path
            + "IS_CS\\snli_with_misleading_score\\snli_with_misleading_rate.jsonl",
            "keys": ("s1", "s2", "gold_label"),
        },  ## mnli-hard match_lambda07 409  ---
        "LI_LI": {
            "path": base_path + "LI_LI_d\\dataset.jsonl",
            "keys": ("sentence1", "sentence2", "gold_label"),
        },  ## snli LI_LI 8193
        "ST": {
            "path": base_path + "ST\\test.jsonl",
            "keys": ("sentence1", "sentence2", "gold_label"),
        },  ## mnli stress(distraction noise) test 104546
        "HANS": {
            "path": base_path + "HANS\\test.jsonl",
            "keys": ("premise", "hypothesis", "gold"),
        },  ##  30000
        "MNLI_hard_m": {
            "path": base_path + "MNLI_hard_m\\test.jsonl",
            "keys": ("sentence1", "sentence2", "gold_label"),
        },  ## mnli_hard_m 4573
        "MNLI_hard_mm": {
            "path": base_path + "MNLI_hard_mm\\test.jsonl",
            "keys": ("sentence1", "sentence2", "gold_label"),
        },  ## mnli_hard_mm 4530
        "QNLI": {
            "path": base_path + "QNLI\\dev.jsonl",
            "keys": ("premise", "hypothesis", "gold"),
        },  ## QNLI 5266
    }
    return task_dict


def read_jsonl(path, mode="r"):
    ls = []
    with open(path, mode, encoding="utf-8") as f:
        for line in f:
            ls.append(json.loads(line))
    if len(ls) == 1:
        ls = ls[0]
    return ls, len(ls)


def read_tsv(tsv_file_path):
    data = pd.read_csv(tsv_file_path, sep="\t")
    data = data.values.tolist()
    json_list = []
    for label, sen1, sen2 in data:
        json_list.append({"gold_label": label, "sentence1": sen1, "sentence2": sen2})
    return json_list, len(json_list)


def read_test(path):
    lines = []
    with open(path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file]
    return lines, len(lines)


def save(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json_line = json.dumps(data)
        file.write(json_line + "\n")
