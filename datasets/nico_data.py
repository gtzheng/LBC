import os
import sys
import re
import datetime

import numpy as np
import json
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
import pandas as pd
from PIL import Image
import pickle
import warnings
from config import NICO_DATA_FOLDER, NICO_CXT_DIC_PATH, NICO_CLASS_DIC_PATH

# reference to https://github.com/Wangt-CN/CaaM and https://github.com/Nayeong-V-Kim/LWBC/tree/master

def prepare_metadata():
    cxt_dic = json.load(open(NICO_CXT_DIC_PATH, 'r'))
    class_dic = json.load(open(NICO_CLASS_DIC_PATH, 'r'))
    cxt_index2name = {i: n for n, i in cxt_dic.items()}
    class_index2name = {i: n for n, i in class_dic.items()}

    labels = []
    contexts = []
    context_names = []
    label_names = []
    file_names = []
    splits = []
    for split_id, split in enumerate(["train", "val", "test"]):
        all_file_name = os.listdir(os.path.join(NICO_DATA_FOLDER, split))
        for file_name in all_file_name:
            label, context, index = file_name.split('_')
            file_names.append(os.path.join(split, file_name))
            contexts.append(int(context))
            context_names.append(cxt_index2name[int(context)])
            label_names.append(class_index2name[int(label)])
            labels.append(int(label))
            splits.append(split_id)

    labels_unique = sorted(list(set(labels)))
    contexts_unique = sorted(list(set(contexts)))
    label2unique = {l: i for i, l in enumerate(labels_unique)}
    context2unique = {c: i for i, c in enumerate(contexts_unique)}
    uniquelabel2name = {
        label2unique[l]: class_index2name[l] for l in labels_unique}
    uniquecontext2name = {
        context2unique[c]: cxt_index2name[c] for c in contexts_unique}

    name2uniquelabel = {n: l for l, n in uniquelabel2name.items()}
    name2uniquecontext = {n: c for c, n in uniquecontext2name.items()}

    with open(os.path.join(NICO_DATA_FOLDER, "metadata.csv"), "w") as f:
        f.write("img_id,img_filename,y,label_name,split,context,context_name\n")
        for i in range(len(file_names)):
            file_name = file_names[i]
            label = label2unique[labels[i]]
            label_name = label_names[i]
            split_id = splits[i]
            context = context2unique[contexts[i]]
            context_name = context_names[i]
            f.write(
                f"{i},{file_name},{label},{label_name},{split_id},{context},{context_name}\n")


def get_transform_nico(train, augment_data=True):
    mean = [0.52418953, 0.5233741, 0.44896784]
    std = [0.21851876, 0.2175944, 0.22552039]
    if train and augment_data:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform


class NICO_dataset(Dataset):
    def __init__(self, basedir, split, balance_factor=1.0, transform=None, attribute_embed=None):
        super(NICO_dataset, self).__init__()
        assert split in ["train", "val", "test"], f"invalida split = {split}"
        self.basedir = basedir
        metadata_df = pd.read_csv(os.path.join("datasets", "sel_metadata.csv"))
        split_info = metadata_df["split"].values
        print(len(metadata_df))
        split_i = ["train", "val", "test"].index(split)
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]

        self.y_array = self.metadata_df["y"].values
        sel_indexes = self.metadata_df["img_id"].values
        labelnames = self.metadata_df["label_name"].values
        self.labelname2index = {}
        for i in range(len(self.y_array)):
            self.labelname2index[labelnames[i]] = self.y_array[i]

        self.p_array = self.metadata_df["context"].values
        contextnames = self.metadata_df["context_name"].values
        self.contextname2index = {}
        for i in range(len(self.p_array)):
            self.contextname2index[contextnames[i]] = self.p_array[i]
        self.filename_array = self.metadata_df["img_filename"].values
        
        self.n_classes = np.unique(self.y_array).size
        self.n_places = np.unique(self.p_array).size

        self.group_array = (
            self.y_array * self.n_places + self.p_array
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

        self.transform = transform

        if attribute_embed and os.path.exists(attribute_embed):
            with open(attribute_embed, "rb") as f:
                self.embeddings = pickle.load(f)
            self.embeddings = self.embeddings[sel_indexes]
            print(len(self.embeddings))
        else:
            self.embeddings = None

   
    def get_group(self, idx):
        y = self.y_array[idx]
        g = (self.embeddings[idx] == 1) * self.n_classes + y
        return g

    def __getitem__(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        y = self.y_array[idx]
        p = self.p_array[idx]
        g = self.group_array[idx]

        if self.embeddings is None:
            return img, y, g, p
        else:
            return img, y, g, p, self.get_group(idx)

    def __len__(self):
        return len(self.y_array)
