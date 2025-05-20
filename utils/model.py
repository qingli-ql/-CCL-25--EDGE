import torch
import os
from module.util import get_model
import numpy as np
import json
import torch.nn as nn

def load_model(cfg, attr_dims, model_path):
    model = get_model(cfg, attr_dims[0]).to(cfg["device"])
    
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def init_model(cfg, attr_dims):
    model = get_model(cfg, attr_dims[0]).to(cfg["device"])
    
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["main_learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    return model, optimizer


def init_pre_models(cfg, attr_dims):
    umodel = load_model(
        cfg,
        attr_dims,
        os.path.join(
            cfg["log_dir"],
            "ColoredMNIST-Skewed0.9-Severity4",
            "Vanilla_seed0MLP",
            "model_last.th",
        ),
    )
    bmodel = load_model(
        cfg,
        attr_dims,
        os.path.join(
            cfg["log_dir"],
            "ColoredMNIST-Skewed0.01-Severity4",
            "bn_seed0MLP",
            "model_last.th",
        ),
    )
    return umodel, bmodel


def save_model(cfg, model, optimizer, epoch, total_acc, type="best"):
    model_path = os.path.join(
        cfg["file_path"],
        f"model_{type}.th",
    )
    state_dict = {
        "steps": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "accs": total_acc,
    }
    torch.save(state_dict, model_path)


def save_file(cfg, data, file_name):
    file_path = os.path.join(
        cfg["file_path"],
        file_name,
    )
    # 写入 JSON 数据到文件
    with open(file_path, "w") as json_file:
        json_file.write(json.dumps(data))
