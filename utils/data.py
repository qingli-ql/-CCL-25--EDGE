import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from data.util import get_dataset
from utils.metric import get_metrics


def init_loader(cfg):
    # prepare train data
    train_dataset = get_dataset(
        cfg["dataset_tag"],
        data_dir=cfg["data_dir"],
        dataset_split="train",
        transform_split="train",
    )
    train_target_attr = train_dataset.attr[:, cfg["target_attr_idx"]]
    train_bias_attr = train_dataset.attr[:, cfg["bias_attr_idx"]]
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # prepare valid data
    valid_dataset = get_dataset(
        cfg["dataset_tag"],
        data_dir=cfg["data_dir"],
        dataset_split="eval",
        transform_split="eval",
    )
    valid_target_attr = valid_dataset.attr[:, cfg["target_attr_idx"]]
    valid_bias_attr = valid_dataset.attr[:, cfg["bias_attr_idx"]]
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataset, train_loader, valid_dataset, valid_loader, attr_dims


@torch.no_grad()
def evaluate(cfg, model, data_loader, show=False, dataset_type="VALID", model_type="MAIN"):
    log = cfg["log"]
    device = cfg["device"]
    model.eval()
    pd_list = list(range(len(data_loader)))
    gt_list = list(range(len(data_loader)))
    type_list = list(range(len(data_loader)))
    for idx, batch in enumerate(data_loader):
        _, data, attr = batch
        data, attr = data.to(device), attr.to(device)

        logit = model(data)
        pred = logit.data.max(1, keepdim=True)[1].squeeze(1)

        pd_list[idx] = pred
        gt_list[idx] = attr[:, cfg["target_attr_idx"]]
        isSkewed = attr[:, cfg["bias_attr_idx"]] == attr[:, cfg["target_attr_idx"]]
        type_list[idx] = isSkewed

    y_pd = torch.hstack(pd_list)
    y_gt = torch.hstack(gt_list)
    y_type = torch.hstack(type_list)
    results = {}
    key2name = {True: "aligned", False: "conflicting"}
    metrics = ["acc"]  # , "loss"]
    for key, name in key2name.items():
        mask = y_type == key
        eval_res = {}
        for metric in metrics:
            eval_res[metric] = get_metrics(metric)(y_gt[mask], y_pd[mask])
        eval_res = {
            "count": mask.sum().item(),
            **eval_res,
        }
        results[name] = eval_res
    # total
    eval_res = {}
    for metric in metrics:
        eval_res[metric] = get_metrics(metric)(y_gt, y_pd)
    eval_res = {
        "count": y_type.shape[0],
        **eval_res,
    }
    results["total"] = eval_res

    if show:
        log.info(f"<<<-------- {dataset_type} dataset on {model_type} model -------->>>")
        for tag in ["total", "aligned", "conflicting"]:
            log.info(
                {
                    f"{tag:>11}": {
                        key: f"{value:6.2f}" for key, value in results[tag].items()
                    }
                }
            )
    return results


@torch.no_grad()
def evaluate_confidence(cfg, model, bmodel,  data_loader, show=False, type="confidence", confidence=0):
    log = cfg["log"]
    device = cfg["device"]
    model.eval()
    bmodel.eval()
    pd_list = list(range(len(data_loader)))
    gt_list = list(range(len(data_loader)))
    type_list = list(range(len(data_loader)))
    for idx, batch in enumerate(data_loader):
        _, data, attr = batch
        data, attr = data.to(device), attr.to(device)
        
        logit = model(data)
        logit_prob = F.softmax(logit, dim=-1)
        logit_confidence = torch.max(logit_prob, dim=-1)[0]
        
        blogit = bmodel(data)
        blogit_prob = F.softmax(blogit, dim=-1)
        blogit_confidence = torch.max(blogit_prob, dim=-1)[0]
        
        if confidence == 0:
            mask = blogit_confidence > logit_confidence
        else:
            mask = blogit_confidence > confidence
        pred = torch.zeros_like(mask, dtype=torch.long)
        pred[mask] = blogit_prob[mask].argmax(dim=-1)
        pred[~mask] = logit_prob[~mask].argmax(dim=-1)
        pd_list[idx] = pred
        gt_list[idx] = attr[:, cfg["target_attr_idx"]]
        isSkewed = attr[:, cfg["bias_attr_idx"]] == attr[:, cfg["target_attr_idx"]]
        type_list[idx] = isSkewed

    y_pd = torch.hstack(pd_list)
    y_gt = torch.hstack(gt_list)
    y_type = torch.hstack(type_list)
    results = {}
    key2name = {True: "aligned", False: "conflicting"}
    metrics = ["acc"]  # , "loss"]
    for key, name in key2name.items():
        mask = y_type == key
        eval_res = {}
        for metric in metrics:
            eval_res[metric] = get_metrics(metric)(y_gt[mask], y_pd[mask])
        eval_res = {
            "count": mask.sum().item(),
            **eval_res,
        }
        results[name] = eval_res
    # total
    eval_res = {}
    for metric in metrics:
        eval_res[metric] = get_metrics(metric)(y_gt, y_pd)
    eval_res = {
        "count": y_type.shape[0],
        **eval_res,
    }
    results["total"] = eval_res

    if show:
        log.info(f"<<<-------- valid dataset on {type} {confidence} model -------->>>")
        for tag in ["total", "aligned", "conflicting"]:
            log.info(
                {
                    f"{tag:>11}": {
                        key: f"{value:6.2f}" for key, value in results[tag].items()
                    }
                }
            )
    return results

plt_dict = {
    "aligned": {
        "color": "#95BCE5",
        "marker": "^",
        "label": "Aligned",
    },
    "conflicting": {
        "color": "#F39DA0",
        "marker": "*",
        "label": "Conflicting",
    },
    "total": {
        "color": "black",
        "marker": "o",
        "label": "Total",
    },
}


def draw_acc(cfg, list_valid):
    metric = "acc"
    metrics = list_valid
    num_epoch = len(metrics["aligned"])

    font_size = 8
    plt.rc("font", family="Times New Roman")
    # plt.rcParams['text.usetex'] = True
    plt.rcParams["lines.linewidth"] = 1
    plt.figure(figsize=(3, 1.5), dpi=300, facecolor="none")
    plt.rcParams.update({"font.size": font_size})
    plt.xlabel("step", fontsize=font_size)
    plt.ylabel("rate", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    for type, _dict in metrics.items():
        plt.plot(
            range(num_epoch)[:num_epoch],
            metrics[type][:num_epoch],
            label=plt_dict[type]["label"],
            color=plt_dict[type]["color"],
            marker=plt_dict[type]["marker"],
            markersize="0",
            linewidth=1,
        )

    # 设置图框线粗细
    bwith = 0.5  # 边框宽度设置为2
    TK = plt.gca()  # 获取边框
    TK.spines["bottom"].set_linewidth(bwith)  # 图框下边
    TK.spines["left"].set_linewidth(bwith)  # 图框左边
    TK.spines["top"].set_linewidth(bwith)  # 图框上边
    TK.spines["right"].set_linewidth(bwith)  # 图框右边

    # 添加标签和标题
    plt.xlabel(f"epoch")
    plt.ylabel(f"ACC")
    plt.xticks(np.arange(0, num_epoch + 1, (num_epoch + 1) // 5))
    plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2))
    # 显示图表
    plt.savefig(
        f"{cfg['file_path']}/{metric}.svg",
        format="svg",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )
    plt.show()


def draw_consine(cfg, list_consines):
    # list_consines 14 * 30
    all_consines = np.array(list_consines).reshape(-1)
    epoch_num = len(list_consines)
    batch_num = len(list_consines[0])
    # 展示前5个epoch的每一个batch的数据
    show_epoch_num = 10
    show_idx_num = batch_num * show_epoch_num
    metric = "theta"
    metrics = all_consines

    font_size = 8
    # plt.rc("font", family="Times New Roman")
    # plt.rcParams['text.usetex'] = True
    plt.rcParams["lines.linewidth"] = 1
    plt.figure(figsize=(3, 1), dpi=300, facecolor="none")
    plt.rcParams.update({"font.size": font_size})
    plt.xlabel("step", fontsize=font_size)
    plt.ylabel("rate", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.plot(
        range(show_idx_num),
        metrics[:show_idx_num],
        label=r"$\theta$",
        color="#95BCE5",
        markersize="0",
        linewidth=1,
    )

    plt.plot(
        range(show_idx_num),
        [90] * show_idx_num,
        color="black",
        markersize="0",
        linewidth=1,
        linestyle="--",
    )

    # 设置图框线粗细
    bwith = 0.5  # 边框宽度设置为2
    TK = plt.gca()  # 获取边框
    TK.spines["bottom"].set_linewidth(bwith)  # 图框下边
    TK.spines["left"].set_linewidth(bwith)  # 图框左边
    TK.spines["top"].set_linewidth(bwith)  # 图框上边
    TK.spines["right"].set_linewidth(bwith)  # 图框右边

    # 添加标签和标题
    plt.xlabel(f"Epoch")
    plt.ylabel(r"$\theta$")

    epoch_xticks = np.arange(0, show_idx_num + 1, (show_idx_num + 1) // show_epoch_num)
    epoch_xlabel = [f"{i}" for i in range(show_epoch_num + 1)]
    plt.xticks(ticks=epoch_xticks, labels=epoch_xlabel)

    plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.3))
    # 显示图表
    plt.savefig(
        f"{cfg['file_path']}/{metric}.svg",
        format="svg",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )
    plt.show()
