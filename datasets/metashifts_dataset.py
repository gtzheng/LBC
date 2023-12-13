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

class MetaShiftsDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None, concept_embed=None):
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
        self.group_array = self.metadata_df["group"].values
        self.n_classes = np.unique(self.y_array).size
        self.n_groups = len(np.unique(self.group_array))
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
        self.filename_array = self.metadata_df["img_filename"].values
        if concept_embed and os.path.exists(concept_embed):
            with open(concept_embed, "rb") as f:
                self.embeddings = pickle.load(f)
            self.embeddings = self.embeddings[split_info==split_i]
            # if split == "val":
            #     self.concepts = self.sel_concepts()
            # else:
            #     self.concepts = None
        else:
            self.embeddings = None
            self.concepts = None
    def __len__(self):
        return len(self.filename_array)

    def get_group(self, idx):
        y = self.y_array[idx]
        g = (self.embeddings[idx]==1) * self.n_classes + y
        # if self.concepts is None:
        #     return -1
        # y = self.y_array[idx]
        # if self.embeddings[idx, self.concepts].sum() == 1:
        #     g = y * len(self.concepts) + np.argmax(self.embeddings[idx, self.concepts])
        # else:
        #     g = -1
        return g
        
    
    def sel_concepts(self):
        sel_concepts = []
        num_samples, num_attrs = self.embeddings.shape
        scores = []
        for i in tqdm(range(num_attrs-1)):
            for j in range(i+1, num_attrs):
                counts = self.embeddings[:,i] + self.embeddings[:,j]
                indexes = np.arange(num_samples)[counts == 1]
                num_i = [self.embeddings[indexes][:,i][self.y_array[indexes] == l].sum() for l in range(self.n_classes)]
                num_j = [self.embeddings[indexes][:,j][self.y_array[indexes] == l].sum() for l in range(self.n_classes)]
                nums = np.array(num_i+num_j)
                if nums.min() < 100:
                    continue
                num_i = np.array(num_i)
                num_j = np.array(num_j)
                prob_i = num_i/num_i.sum()
                ent_i = (-prob_i * np.log(prob_i+1e-10)).sum()
                prob_j = num_j / num_j.sum()
                # ent_j = (-prob_j*np.log(prob_j+1e-10)).sum()
                dist = np.linalg.norm(num_i - num_j, 2.0)
                scores.append(((i,j),ent_i * dist))
        scores = sorted(scores, key=lambda x:x[1])
        return np.array(scores[0][0])
    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")
        # img = read_image(img_path)
        # img = img.float() / 255.

        if self.transform:
            img = self.transform(img)
        
        # if self.segmask:
        #     img_path = os.path.join(
        #         self.segmask, self.filename_array[idx].replace(".jpg", ".png")
        #     )
        #     seg = Image.open(img_path).convert("RGB")
        #     seg = self.transform(seg)
        #     return img, y, g, p, seg
        # else:
        if self.embeddings is None:
            return img, y, g, g
        else:
            return img, y, g, g, self.get_group(idx)


def get_transform_cub(target_resolution, train, augment_data):
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


def get_loader(
    data, train, reweight_groups, reweight_classes, reweight_places, **kwargs
):
    if not train:  # Validation or testing
        assert reweight_groups is None
        assert reweight_classes is None
        assert reweight_places is None
        shuffle = False
        sampler = None
    elif not (
        reweight_groups or reweight_classes or reweight_places
    ):  # Training but not reweighting
        shuffle = True
        sampler = None
    elif reweight_groups:
        # Training and reweighting groups
        # reweighting changes the loss function from the normal ERM (average loss over each training example)
        # to a reweighted ERM (weighted average where each (y,c) group has equal weight)
        group_weights = len(data) / data.group_counts
        weights = group_weights[data.group_array]

        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    elif reweight_classes:  # Training and reweighting classes
        class_weights = len(data) / data.y_counts
        weights = class_weights[data.y_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    else:  # Training and reweighting places
        place_weights = len(data) / data.p_counts
        weights = place_weights[data.p_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False

    loader = DataLoader(data, shuffle=shuffle, sampler=sampler, **kwargs)
    return loader


def log_data(logger, train_data, test_data, val_data=None, get_yp_func=None):
    logger.write(f"Training Data (total {len(train_data)})\n")
    # group_id = y_id * n_places + place_id
    # y_id = group_id // n_places
    # place_id = group_id % n_places
    for group_idx in range(train_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(
            f"    Group {group_idx} (y={y_idx}, p={p_idx}): n = {train_data.group_counts[group_idx]:.0f}\n"
        )
    logger.write(f"Test Data (total {len(test_data)})\n")
    for group_idx in range(test_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(
            f"    Group {group_idx} (y={y_idx}, p={p_idx}): n = {test_data.group_counts[group_idx]:.0f}\n"
        )
    if val_data is not None:
        logger.write(f"Validation Data (total {len(val_data)})\n")
        for group_idx in range(val_data.n_groups):
            y_idx, p_idx = get_yp_func(group_idx)
            logger.write(
                f"    Group {group_idx} (y={y_idx}, p={p_idx}): n = {val_data.group_counts[group_idx]:.0f}\n"
            )
