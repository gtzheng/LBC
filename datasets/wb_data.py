import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle
from tqdm import tqdm

# https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
# https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
# celeba_metadata: https://github.com/PolinaKirichenko/deep_feature_reweighting


class BiasedDataset(Dataset):
    def __init__(
        self,
        basedir: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
        attribute_embed: str = None,
    ):
        """Initialize the BiasedDataset for Waterbirds/CelebA

        Args:
            basedir (str): the base directory of the dataset
            split (str, optional): the split of the dataset. Defaults to "train".
            transform (torchvision.transforms.Compose, optional): the transform for the dataset. Defaults to None.
            attribute_embed (str, optional): the path to the attribute embedding. Defaults to None.
        """
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise (f"Unknown split {split}")
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        split_info = metadata_df["split"].values
        print(len(metadata_df))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        print(len(self.metadata_df))
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        self.p_array = self.metadata_df["place"].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df["place"].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (
            self.y_array * self.n_places + self.confounder_array
        ).astype("int")
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
            (
                torch.arange(self.n_groups).unsqueeze(1)
                == torch.from_numpy(self.group_array)
            )
            .sum(1)
            .float()
        )
        self.y_counts = (
            (
                torch.arange(self.n_classes).unsqueeze(1)
                == torch.from_numpy(self.y_array)
            )
            .sum(1)
            .float()
        )
        self.p_counts = (
            (torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array))
            .sum(1)
            .float()
        )
        self.filename_array = self.metadata_df["img_filename"].values
        if attribute_embed and os.path.exists(attribute_embed):
            with open(attribute_embed, "rb") as f:
                self.embeddings = pickle.load(f)
            self.embeddings = self.embeddings[split_info == split_i]
        else:
            self.embeddings = None

    def __len__(self):
        return len(self.filename_array)

    def get_group(self, idx: int):
        """Get the pseudo group of the image

        Args:
            idx (int): the index of the image.

        Returns:
            int: the pseudo group of the image.
        """
        y = self.y_array[idx]
        g = (self.embeddings[idx] == 1) * self.n_classes + y
        return g

    def __getitem__(self, idx: int):
        """Get the image, label, group, context, and pseudo group of the image (if embeddings is not None)

        Args:
            idx (int): the index of the image

        Returns:
            torch.Tensor: the image
            int: the label of the image
            int: the group of the image
            int: the context of the image
            int: the pseudo group of the image if embeddings is not None
        """
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.embeddings is None:
            return img, y, g, p, p
        else:
            attributes = self.embeddings[idx].astype(float)
            return img, y, g, attributes, self.get_group(idx)


def get_transform_biased(
    target_resolution: list[int, int], train: bool, augment_data: bool
):
    """Get the transform for the Waterbirds/CelebA dataset

    Args:
        target_resolution (list[int, int]): the target resolution of the image
        train (bool): whether the transform is for training
        augment_data (bool): whether the data is augmented

    Returns:
        torchvision.transforms.Compose: the transform for the dataset
    """
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
