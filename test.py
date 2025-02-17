import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
import logging
import os
import argparse
import copy
from models.resnet import resnet18, resnet50
from utils import set_gpu, get_free_gpu
from datasets.data_utils import get_dataloader


def test_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    conflicting_groups: list[int] = None,
):
    """Evaluate the model on the test set

    Args:
        model (torch.nn.Module): the model to evaluate
        loader (torch.utils.data.DataLoader): the test set loader
        conflicting_groups (list[int], optional): the conflicting groups. Defaults to None.

    Returns:
        float: the average accuracy
        float: the unbiased accuracy
        float: the worst-group accuracy
    """
    count = 0
    acc = 0
    model.eval()
    res = []
    groups = []
    with torch.no_grad():
        for x, y, g, p, _ in loader:
            x, y = (
                x.cuda(),
                y.cuda(),
            )
            out = model(x)
            pred = (torch.argmax(out, dim=-1) == y).detach().cpu().numpy()
            res.append(pred)
            groups.append(g.detach().cpu().numpy())
    res = np.concatenate(res)
    avg_acc = res.sum() / len(res)

    groups = np.concatenate(groups, axis=0)
    if groups.ndim == 1:
        num_group_types = 1
        groups = groups.reshape(-1, 1)
    else:
        num_group_types = groups.shape[1]
    unbiased_acc_avg = 0
    worst_acc_avg = 0

    for g_id in range(num_group_types):
        acc_group = []
        group_num = []

        unique_groups = np.unique(groups[:, g_id])
        group2idx = {g: i for i, g in enumerate(unique_groups)}
        for g in unique_groups:
            gres = res[groups[:, g_id] == g]
            if len(gres) < 10:
                continue
            acc_group.append(gres.sum() / len(gres))
            group_num.append(len(gres))
        acc_group = np.array(acc_group)
        unbiased_acc_avg += acc_group.mean()
        worst_acc_avg += acc_group.min()

    unbiased_acc_avg /= num_group_types
    worst_acc_avg /= num_group_types
    return avg_acc, unbiased_acc_avg, worst_acc_avg


def test_model_pseudo(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, num_threshold: int = 20
):
    """Evaluate the model on the test set using pseudo group labels

    Args:
        model (torch.nn.Module): the model to evaluate
        loader (torch.utils.data.DataLoader): the test set loader
        num_threshold (int, optional): the threshold for the number of samples in a group. Defaults to 20.

    Returns:
        float: the average accuracy
        float: the worst-pseudo-group accuracy
        float: the average-pseudo-group accuracy
    """
    count = 0
    acc = 0
    model.eval()
    res = []
    groups_psu = []
    with torch.no_grad():
        for x, y, _, p, g_arr in loader:
            x, y = (
                x.cuda(),
                y.cuda(),
            )
            out = model(x)
            pred = (torch.argmax(out, dim=-1) == y).detach().cpu().numpy()
            res.append(pred)
            groups_psu.append(g_arr.detach().cpu().numpy())
    groups_psu = np.concatenate(groups_psu)
    res = np.concatenate(res)

    attr_worst_acc = []
    attr_avg_acc = []
    for a in range(groups_psu.shape[1]):
        acc_group = []
        group_num = []
        groups = groups_psu[:, a]
        uni_groups = np.unique(groups)
        n_groups = len(uni_groups)
        for g in range(
            n_groups // 2, n_groups
        ):  # select the group that has the attribute
            gres = res[groups == g]
            if len(gres) > num_threshold:
                acc_group.append(gres.sum() / len(gres))
                group_num.append(len(gres))
        if len(acc_group) > 0:
            acc_group = np.array(acc_group)
            worst_acc_psu = acc_group.min()
            attr_worst_acc.append(worst_acc_psu)
            attr_avg_acc.append(acc_group)
    attr_worst_acc = np.array(attr_worst_acc)
    attr_avg_acc = np.concatenate(attr_avg_acc)
    avg_acc = res.sum() / len(res)

    return avg_acc, attr_worst_acc.min(), attr_avg_acc.mean()


def load_model(ckpt_path: str):
    """Load a model from a checkpoint

    Args:
        ckpt_path (str): the path to the checkpoint

    Returns:
        torchvision.models.resnet.ResNet: the model
    """
    model = torch.load(ckpt_path)
    model.cuda()
    model.eval()
    return model


class LBC(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        n_clusters: int = 2,
        pretrained: bool = True,
    ):
        """Initialize the LBC model

        Args:
            backbone (str): the backbone model
            num_classes (int): the number of classes
            n_clusters (int, optional): the number of clusters. Defaults to 2.
            pretrained (bool, optional): whether to use the pretrained model. Defaults to True.
        """
        super(LBC, self).__init__()
        if backbone == "resnet50":
            if pretrained:
                self.backbone = resnet50()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet50_Weights.DEFAULT.get_state_dict(
                        progress=True
                    ),
                    strict=False,
                )
            else:
                self.backbone = resnet50()
        elif backbone == "resnet18":
            if pretrained:
                self.backbone = resnet18()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet18_Weights.DEFAULT.get_state_dict(
                        progress=True
                    ),
                    strict=False,
                )
            else:
                self.backbone = resnet18()
        d = self.backbone.out_dim

        self.classifier = nn.Linear(d, num_classes * n_clusters)
        self.num_classes = num_classes
        self.fea_dim = d
        self.fc = nn.Linear(d, num_classes)
        self.K = n_clusters

    def normal_forward(self, x: torch.tensor):
        """Normal prediction function

        Args:
            x (torch.tensor): input image

        Returns:
            torch.tensor: the logits
        """
        fea = self.backbone(x)
        logits = self.fc(fea)
        return logits

    def forward(self, x: torch.tensor, pred: bool = False):
        """LBC forward function

        Args:
            x (torch.tensor): input image
            pred (bool, optional): whether to return the prediction. Defaults to False.

        Returns:
            if is in training mode:
                torch.tensor: the logits (fine-grained)
                torch.tensor: the prediction if pred is True
            else:
                torch.tensor: the class logits
        """
        fea = self.backbone(x)
        logits = self.classifier(fea)
        if self.classifier.training:
            if pred:
                preds = torch.argmax(logits, dim=1)
                preds = preds // self.K
                return logits, preds
            else:
                return logits
        else:
            class_logits = torch.max(
                logits.reshape(-1, self.num_classes, self.K), dim=-1
            )[0]
            return class_logits


class ERMModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True):
        """Initialize the ERM model

        Args:
            backbone (str): the backbone architecture
            num_classes (int): number of classes
            pretrained (bool, optional): whether to use pretrained model. Defaults to True.
        """
        super(ERMModel, self).__init__()
        if backbone == "resnet50":
            if pretrained:
                self.backbone = resnet50()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet50_Weights.DEFAULT.get_state_dict(
                        progress=True
                    ),
                    strict=False,
                )
            else:
                self.backbone = resnet50()
        elif backbone == "resnet18":
            if pretrained:
                self.backbone = resnet18()
                self.backbone.load_state_dict(
                    torchvision.models.ResNet18_Weights.DEFAULT.get_state_dict(
                        progress=True
                    ),
                    strict=False,
                )
            else:
                self.backbone = resnet18()
        d = self.backbone.out_dim

        self.num_classes = num_classes
        self.fea_dim = d
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        fea = self.backbone(x)
        logits = self.fc(fea)
        return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lbc test")

    parser.add_argument(
        "--dataset",
        default="waterbirds",
        type=str,
        help="select dataset",
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--model_type",
        default="",
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--model_path",
        default="",
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--n_clusters",
        default=3,
        type=int,
        help="number of clusters",
    )
    args = parser.parse_args()
    train_loader, idx_train_loader, val_loader, test_loader = get_dataloader(
        args.dataset, args.batch_size
    )
    if args.dataset == "waterbirds":
        num_classes = 2
    elif args.dataset == "celeba":
        num_classes = 2
    elif args.dataset == "nico":
        num_classes = 10
    elif args.dataset == "imagenet-9":
        num_classes = 9
    if args.model_type == "erm":
        model = ERMModel(args.backbone, num_classes)
    elif args.model_type == "lbc":
        model = LBC(args.backbone, num_classes, args.n_clusters)
    gpu = ",".join([str(i) for i in get_free_gpu()[0:1]])
    set_gpu(gpu)
    model_dict = torch.load(args.model_path)
    model.load_state_dict(model_dict, strict=False)
    model.backbone.load_state_dict(model_dict, strict=False)

    model.cuda()
    avg_acc, unbiased_acc, worst_acc = test_model(model, test_loader)
    print(
        f"Avg acc: {avg_acc:.6f}, unbiased_acc {unbiased_acc:.6f}, worst acc: {worst_acc:.6f}"
    )
