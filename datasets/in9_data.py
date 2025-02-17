import json
import os
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import pickle
from tqdm import tqdm
import warnings

from torchvision.datasets import ImageFolder

# Command to download the ImageNet dataset
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
# reference to https://github.com/clovaai/rebias/blob/master/datasets/imagenet.py

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]
# 0: "dog", 1: "cat", 2: "frog", 3: "turtle", 4: "bird", 5: "primate", 6: "fish", 7: "crab", 8: "insect"
CLASS_TO_INDEX = {
    "n01641577": 2,
    "n01644373": 2,
    "n01644900": 2,
    "n01664065": 3,
    "n01665541": 3,
    "n01667114": 3,
    "n01667778": 3,
    "n01669191": 3,
    "n01819313": 4,
    "n01820546": 4,
    "n01833805": 4,
    "n01843383": 4,
    "n01847000": 4,
    "n01978287": 7,
    "n01978455": 7,
    "n01980166": 7,
    "n01981276": 7,
    "n02085620": 0,
    "n02099601": 0,
    "n02106550": 0,
    "n02106662": 0,
    "n02110958": 0,
    "n02123045": 1,
    "n02123159": 1,
    "n02123394": 1,
    "n02123597": 1,
    "n02124075": 1,
    "n02174001": 8,
    "n02177972": 8,
    "n02190166": 8,
    "n02206856": 8,
    "n02219486": 8,
    "n02486410": 5,
    "n02487347": 5,
    "n02488291": 5,
    "n02488702": 5,
    "n02492035": 5,
    "n02607072": 6,
    "n02640242": 6,
    "n02641379": 6,
    "n02643566": 6,
    "n02655020": 6,
}


def is_image_file(filename: str):
    """
    Determine whether the given filename is an image
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def collect_dataset_info(
    dir: str, class_to_idx: dict[str, int], dataset: str = "ImageNet-9"
):
    """Get image filenames and the mapping from class names to integers for ImageNet-9 or ImageNet-A

    Args:
        dir (str): path to the dataset folder
        class_to_idx (dict[str, int]): a dictionary that maps class names to integers
        dataset (str, optional): choose from {ImageNet, ImageNet-A}

    Returns:
        Tuple[List[str], Dict[str, int]]: a list of image filenames and an updated dictionary containing the mapping from class names to integers.
    """
    # dog, cat, frog, turtle, bird, monkey, fish, crab, insect
    RESTRICTED_RANGES = [
        (151, 254),
        (281, 285),
        (30, 32),
        (33, 37),
        (89, 97),
        (372, 378),
        (393, 397),
        (118, 121),
        (306, 310),
    ]
    range_sets = [set(range(s, e + 1)) for s, e in RESTRICTED_RANGES]
    class_to_idx_ = {}

    if dataset == "ImageNet-A":
        for class_name, idx in class_to_idx.items():
            try:
                class_to_idx_[class_name] = CLASS_TO_INDEX[class_name]
            except Exception:
                pass
    elif dataset == "ImageNet-9":  # ImageNet
        for class_name, idx in class_to_idx.items():
            for new_idx, range_set in enumerate(range_sets):
                if idx in range_set:
                    if new_idx == 0:  # classes that overlap with ImageNet-A
                        if idx in [151, 207, 234, 235, 254]:
                            class_to_idx_[class_name] = new_idx
                    elif new_idx == 4:
                        if idx in [89, 90, 94, 96, 97]:
                            class_to_idx_[class_name] = new_idx
                    elif new_idx == 5:
                        if idx in [372, 373, 374, 375, 378]:
                            class_to_idx_[class_name] = new_idx
                    else:
                        class_to_idx_[class_name] = new_idx
    images = []
    dir = os.path.expanduser(dir)
    a = sorted(class_to_idx_.keys())
    for target in a:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    item = (os.path.join(target, fname), target, class_to_idx_[target])
                    images.append(item)

    return images, class_to_idx_


def find_classes(folder: str):
    """Get all the class names from the give folder

    Args:
        folder (str): path to an image folder

    Returns:
        Tuple[List[str], Dict[str, int]]: a list of class names and a dictionary containing the mapping from class names to integers
    """
    classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def prepare_imagenet9_metadata(base_dir: str):
    """Create the metadata for the ImageNet-9 dataset and store the results into base_dir/metadata.csv
    The first row is the header: img_id,img_filename,y,label_name,split
    After the header, each row in metadata.csv contains image_id, img_filename, label_id, label_name, and split_id (0 for train and 1 for val)

    Args:
        base_dir (str): the path to the ImageNet dataset (ImageNet-9 is created from ImageNet)

    Returns:
        int: total number of images processed
    """
    with open(os.path.join(base_dir, "metadata.csv"), "w") as f:
        f.write("img_id,img_filename,y,label_name,split\n")
        image_id = 0
        for split_id, split in enumerate(["train", "val"]):
            data_root = os.path.join(base_dir, split)
            classes, class_to_idx = find_classes(data_root)
            image_info, _ = collect_dataset_info(
                data_root, class_to_idx, dataset="ImageNet-9"
            )

            for idx, info in enumerate(image_info):
                path, target, class_id = info
                path = os.path.join(split, path)
                f.write(f"{image_id},{path},{class_id},{target},{split_id}\n")
                image_id += 1
    return image_id


def prepare_imageneta_metadata(base_dir: str):
    """Create the metadata for the ImageNet-A dataset and store the results into base_dir/metadata.csv
    The first row is the header: img_id,img_filename,y,label_name,split
    After the header, each row in metadata.csv contains image_id, img_filename, label_id, label_name, and split_id (0 for train and 1 for val)

    Args:
        base_dir (str): the path to the ImageNet dataset (ImageNet-9 is created from ImageNet)

    Returns:
        int: total number of images processed
    """
    with open(os.path.join(base_dir, "metadata.csv"), "w") as f:
        f.write("img_id,img_filename,y,label_name\n")
        image_id = 0

        classes, class_to_idx = find_classes(base_dir)
        image_info, _ = collect_dataset_info(
            base_dir, class_to_idx, dataset="ImageNet-A"
        )

        for idx, info in enumerate(image_info):
            path, target, class_id = info
            f.write(f"{image_id},{path},{class_id},{target}\n")
            image_id += 1
    return image_id


class ImageNet9(torch.utils.data.Dataset):
    def __init__(
        self,
        basedir: str,
        split: str,
        transform: torchvision.transforms.Compose = None,
        attribute_embed: str = None,
        cluster_file: str = None,
    ):
        """Initialize the ImageNet-9 dataset

        Args:
            basedir (str): path to the dataset folder
            split (str): choose from {"train", "val", "test"}
            transform (torchvision.transforms.Compose): a series of image transformations
            attribute_embed (str, optional): path to the attribute embeddings. Defaults to None.
            cluster_file (str, optional): path to the cluster file. Defaults to None.
        """
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        split_info = metadata_df["split"].values
        print(len(metadata_df))
        split_i = ["train", "val", "test"].index(split)
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        print(len(self.metadata_df))

        self.y_array = self.metadata_df["y"].values
        self.filename_array = self.metadata_df["img_filename"].values
        self.n_classes = np.unique(self.y_array).size

        self.basedir = basedir
        self.transform = transform

        self.split = split
        if split == "val":
            val_cluster_df = pd.read_csv(cluster_file)

            def add_val_func(x):
                return os.path.join("val", x)

            val_cluster_df["path"] = val_cluster_df["path"].apply(add_val_func)
            val_cluster_df = val_cluster_df.rename(columns={"path": "img_filename"})
            self.metadata_df = self.metadata_df.merge(val_cluster_df, on="img_filename")
            self.p_arrays = [
                self.metadata_df["cluster1"].values,
                self.metadata_df["cluster2"].values,
                self.metadata_df["cluster3"].values,
            ]
            self.n_places = len(np.unique(self.p_arrays[0]))
            self.group_arrays = [
                (self.y_array * self.n_places + self.p_arrays[i]).astype("int")
                for i in range(3)
            ]
            self.n_groups = self.n_classes * self.n_places

        if attribute_embed and os.path.exists(attribute_embed):
            with open(attribute_embed, "rb") as f:
                self.embeddings = pickle.load(f)
                self.embeddings = self.embeddings[split_info == split_i]
        else:
            self.embeddings = None

    def get_group(self, idx: int):
        """Get the pseudo group for the given index

        Args:
            idx (int): index of the image

        Returns:
            int: the pseudo group for the given index.
        """
        y = self.y_array[idx]
        g = (self.embeddings[idx] == 1) * self.n_classes + y
        return g

    def __getitem__(self, index: int):
        """Return the image, target, groups, target (place holder), and pseudo group for the given index

        Args:
            index (int): index of the image

        Returns:
            Tuple[torch.Tensor, int, np.ndarray, int, int]: image, target, groups, target, pseudo group
        """
        path, target = self.filename_array[index], self.y_array[index]
        img_path = os.path.join(self.basedir, path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.split == "val":
            groups = np.array([self.group_arrays[i][index] for i in range(3)])
        else:
            groups = target
        if self.embeddings is None:
            return img, target, groups, target, target
        else:
            return img, target, groups, target, self.get_group(index)

    def __len__(self):
        return len(self.y_array)


class ImageNetA(torch.utils.data.Dataset):
    def __init__(
        self,
        basedir: str,
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the ImageNet-A dataset.

        Args:
            basedir (str): path to the dataset folder
            transform (torchvision.transforms.Compose): a series of image transformations
        """
        self.metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        print(len(self.metadata_df))

        self.y_array = self.metadata_df["y"].values
        self.filename_array = self.metadata_df["img_filename"].values
        self.n_classes = np.unique(self.y_array).size

        self.basedir = basedir
        self.transform = transform

    def __getitem__(self, index: int):
        """Get the image and target for the given index

        Args:
            index (int): index of the image

        Returns:
            Tuple[torch.Tensor, int, int, int, int]: image and target, target x 3 (place holder)
        """
        path, target = self.filename_array[index], self.y_array[index]
        img_path = os.path.join(self.basedir, path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, target, target, target, target

    def __len__(self):
        return len(self.y_array)


def get_imagenet_transform(train: bool, augment_data: bool = True):
    """Get the image transformation for the ImageNet dataset

    Args:
        train (bool): whether the transformation is for training or testing
        augment_data (bool, optional): whether to augment the data. Defaults to True.

    Returns:
        torchvision.transforms.Compose: a series of image transformations
    """
    if train and augment_data:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
    return transform
