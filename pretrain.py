import torch
import torchvision

import numpy as np
import os
import tqdm
import argparse
import json
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from datasets.data_utils import get_dataloader
import torch.nn as nn
from utils import (
    set_gpu,
    get_free_gpu,
    set_log_path,
    log,
    BestMetricGroup,
    Timer,
    time_str,
    AverageMeter,
    set_seed,
)

from config import *
from datetime import datetime

from models.resnet import resnet18, resnet50
from timm.scheduler import create_scheduler


def get_args():
    parser = argparse.ArgumentParser(description="Pretrain models")
    parser.add_argument(
        "--dataset",
        type=str,
        help="waterbirds, celeba, nico",
    )

    parser.add_argument(
        "--backbone", type=str, default="resnet50", help="model architecture"
    )
    parser.add_argument(
        "--pretrained_model", action="store_true", help="Use pretrained model"
    )

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    return args


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


def prepare_model(dataset: str, backbone: str, pretrained: bool = True):
    """Prepare the ERM model

    Args:
        dataset (str): dataset name
        backbone (str): backbone architecture
        pretrained (bool, optional): whether to use pretrained model. Defaults to True.

    Returns:
        ERMModel: the ERM model
    """
    if pretrained:
        print("Use ImageNet pretrained model")
    else:
        print("Train from scratch")
    if dataset == "waterbirds":
        model = ERMModel(backbone, 2, pretrained)
    if dataset == "celeba":
        model = ERMModel(backbone, 2, pretrained)
    if dataset == "nico":
        model = ERMModel(backbone, 10, pretrained)
    if dataset == "imagenet-9":
        model = ERMModel(backbone, 9, pretrained)
    model.cuda()
    return model


def pretrain(args):
    gpu = ",".join([str(i) for i in get_free_gpu()[0:1]])
    set_gpu(gpu)
    now = datetime.now()
    timestamp = now.strftime("%m%d%Y-%H%M%S")
    output_dir = os.path.join(
        EXPR_PATH,
        f"pretrain_{args.dataset}_{args.backbone}_{timestamp}_{os.uname()[1]}",
    )
    print(f"Preparing directory {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if args.dataset == "imagenet-9" and not args.pretrained_model:
        args.init_lr = 0.05
        args.num_epochs = 120
        args.batch_size = 256
        milestones = [50, 80, 100]
        args.weight_decay = 5e-4
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        args_json = json.dumps(vars(args))
        f.write(args_json)

    set_seed(args.seed)
    set_log_path(output_dir)

    # prepare data loaders
    train_loader, idx_train_loader, val_loader, test_loader = get_dataloader(
        args.dataset, args.batch_size
    )
    # prepare the model
    model = prepare_model(args.dataset, args.backbone, pretrained=args.pretrained_model)

    if args.resume is not None:
        print("Resuming from checkpoint at {}...".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)

    model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.init_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    if args.dataset == "imagenet-9":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.2
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )

    criterion = torch.nn.CrossEntropyLoss()

    # Train loop
    best_val = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch in tqdm.tqdm(train_loader, leave=False):
            x, y, _, _, _ = batch
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            acc = (torch.argmax(logits, dim=-1) == y).sum() / len(y)
            acc_meter.update(acc.item(), len(y))

        scheduler.step()
        train_loss_avg = loss_meter.avg
        train_acc_avg = acc_meter.avg

        if epoch % args.eval_freq == 0:
            model.eval()
            acc_meter = AverageMeter()
            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader, leave=False):
                    x, y, _, _, _ = batch
                    x, y = x.cuda(), y.cuda()
                    logits = model(x)
                    acc = (torch.argmax(logits, dim=-1) == y).sum() / len(y)
                    acc_meter.update(acc.item(), len(y))
                if acc_meter.avg > best_val:
                    best_val = acc_meter.avg
                    torch.save(
                        model.state_dict(), os.path.join(output_dir, "best_model.pt")
                    )
        log(
            f"Epoch {epoch}\t Loss: {train_loss_avg:.6f} Acc: {train_acc_avg:.6f} | ValAcc {acc_meter.avg:.6f}\n"
        )
        torch.save(model.state_dict(), os.path.join(output_dir, "latest_model.pt"))


if __name__ == "__main__":
    args = get_args()
    pretrain(args)
