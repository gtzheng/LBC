import json
import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle
from tqdm import tqdm


from torchvision.datasets import ImageFolder


class ImageNet9(ImageFolder):
    def __init__(self, root, dataset, transform=None):
        assert dataset in [
            "original",
            "mixed_rand",
            "mixed_same",
        ]
        super().__init__(
            root=os.path.join(root, f"imagenet-9/{dataset}/val"),
            transform=transform,
        )

        with open(os.path.join(root, "imagenet-9/in_to_in9.json"), "r") as f:
            in_to_in9 = json.load(f)

        new_in_to_in9 = torch.ones(1000, dtype=torch.long) * -1
        for in_idx, in9_idx in in_to_in9.items():
            new_in_to_in9[int(in_idx)] = in9_idx

        self.indices_in_1k = new_in_to_in9

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        data_dict = {
            "image": image,
            "target": target,
        }

        return data_dict

    def map_prediction(self, pred):
        mapped_pred = self.indices_in_1k[pred.to(self.indices_in_1k.device)]
        return mapped_pred


def get_transform_in(target_resolution, train, augment_data):
    scale = 256.0 / 224.0

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        int(target_resolution[0] * scale),
                        int(target_resolution[1] * scale),
                    )
                ),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transform


if __name__ == "__main__":
    dataset = ImageNet9(
        root="/bigtemp/gz5hp/few_shot_dataset",
        dataset="original",
    )

    print(len(dataset))
    print(dataset.__getitem__(2))