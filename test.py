import torchvision
import torch
import numpy as np
# from pytorch_grad_cam import XGradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
import logging
import os

import copy



def set_gpu(gpu):
    print("set gpu:", gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp")
    memory_available = np.array(
        [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    )
    return np.argsort(memory_available)


def test_model(model, loader):
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
    groups = np.concatenate(groups)
    avg_acc = res.sum() / len(res)
    acc_group = []
    group_num = []
    for g in np.unique(groups):
        gres = res[groups == g]
        acc_group.append(gres.sum() / len(gres))
        group_num.append(len(gres))
    acc_group = np.array(acc_group)
    worst_acc = acc_group.min()
    return avg_acc, worst_acc, acc_group


def load_model(n_classes: int, ckpt_path: str) -> torchvision.models.resnet.ResNet:
    # model = torchvision.models.resnet50(weights=None)
    # d = model.fc.in_features
    # model.fc = torch.nn.Linear(d, n_classes)
    model = torch.load(ckpt_path)
    model.cuda()
    model.eval()
    return model



# if __name__ == '__main__':
#     test_transform = get_transform_cub(
#         target_resolution=(224, 224), train=False, augment_data=False
#     )
#     testset = WaterBirdsDataset(
#         basedir="/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2",
#         split="test",
#         transform=test_transform,
#         segmask="/bigtemp/gz5hp/dataset_hub/cub200_2011/CUB_200_2011/segmentations",
#     )

#     loader_kwargs = {
#         "batch_size": 100,
#         "num_workers": 4,
#         "pin_memory": True,
#         "reweight_places": None,
#     }

#     test_loader = get_loader(
#         testset,
#         train=False,
#         reweight_groups=None,
#         reweight_classes=None,
#         **loader_kwargs,
#     )


#     gpu = ",".join([str(i) for i in get_freer_gpu()[0:1]])
#     set_gpu(gpu)

#     model_path = "/bigtemp/gz5hp/spurious_correlations/mask_expr/model_gval_npc3.pt"
#     model = load_model(2, model_path)
#     avg_acc, worst_acc = test_model(model, test_loader)
#     print(f"Avg acc: {avg_acc:.4f}, worst acc: {worst_acc:.4f}")
