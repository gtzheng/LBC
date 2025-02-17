from datasets.wb_data import (
    BiasedDataset,
    get_transform_biased,
)
from datasets.nico_data import (
    get_transform_nico,
    NICO_dataset,
)

from datasets.in9_data import (
    get_imagenet_transform,
    ImageNet9,
    ImageNetA,
)
from config import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler


class IdxDataset(Dataset):
    """
    A dataset wrapper which returns the corresponding indexes along with items in the dataset
    """

    def __init__(self, dataset: Dataset):
        """
        initialize with dataset
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        return the idx'th item from self.dataset along with the index
        """
        return (idx, *self.dataset[idx])


def get_waterbirds_loader(batch_size: int, img_size: tuple[int, int] = (224, 224)):
    """Return train_loader, idx_train_loader, val_loader, and test_loader for the Waterbirds dataset.
    Require the following constant specified in config.py:
        - WATERBIRDS_ATTRIBUTE_PATH: path to the attributes extracted for the dataset. Attributes can be extracted using extract_attributes.py
        - WATERBIRDS_DATA_FOLDER: path to the Waterbirds dataset.
    Only train_loader will return augmented samples.
    The idx_train_loader will return original training samples along with their indexes in the training dataset.

    Args:
        batch_size (int): batch size
        img_size (tuple[int, int], optional): the width and height of input images. Default to (224, 224)

    Returns:
        tuple[DataLoader]: train_loader, idx_train_loader, val_loader, test_loader
    """
    # get data transformations with (train_transform) and without (test_transform) data augmentations
    train_transform = get_transform_biased(
        target_resolution=img_size, train=True, augment_data=True
    )
    test_transform = get_transform_biased(
        target_resolution=img_size, train=False, augment_data=False
    )

    attribute_path = WATERBIRDS_ATTRIBUTE_PATH
    trainset = BiasedDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="train",
        transform=train_transform,
        attribute_embed=attribute_path,
    )
    trainset_ref = BiasedDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="train",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    idx_train_loader = DataLoader(
        train_idx_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    valset = BiasedDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="val",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    testset = BiasedDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="test",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    return train_loader, idx_train_loader, val_loader, test_loader


def get_celeba_loader(
    batch_size: int, img_size: tuple[int, int] = (224, 224), sampling: bool = True
):
    """Return train_loader, idx_train_loader, val_loader, and test_loader for the CelebA dataset
    Require the following constant specified in config.py:
        - CELEBA_ATTRIBUTE_PATH: path to the attributes extracted for the dataset. Attributes can be extracted using extract_attributes.py
        - CELEBA_DATA_FOLDER: path to the CelebA dataset
        - NUM_BATCHES: number of batches to sample per epoch if sampling is True
    Only train_loader will return augmented samples
    The idx_train_loader will return original training samples along with their indexes in the training dataset

    Args:
        batch_size (int): batch size
        img_size (tuple[int, int], optional): the width and height of input images. Default to (224, 224).
        sampling (bool, optional): set whether to sample a portion of training data to reduce training time.

    Returns:
        tuple[DataLoader]: train_loader, idx_train_loader, val_loader, test_loader
    """

    # get data transformations with (train_transform) and without (test_transform) data augmentations
    train_transform = get_transform_biased(
        target_resolution=img_size, train=True, augment_data=True
    )
    test_transform = get_transform_biased(
        target_resolution=img_size, train=False, augment_data=False
    )

    attribute_path = CELEBA_ATTRIBUTE_PATH

    trainset = BiasedDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="train",
        transform=train_transform,
        attribute_embed=attribute_path,
    )
    trainset_ref = BiasedDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="train",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    if sampling == True:
        idx_train_loader = DataLoader(
            train_idx_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(
                train_idx_dataset, num_samples=batch_size * NUM_BATCHES
            ),
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
    else:
        idx_train_loader = DataLoader(
            train_idx_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
    valset = BiasedDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="val",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    testset = BiasedDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="test",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    return train_loader, idx_train_loader, val_loader, test_loader


def get_nico_loader(batch_size: int):
    """Return train_loader, idx_train_loader, val_loader, and test_loader for the NICO dataset
    Require the following constant specified in config.py:
        - NICO_ATTRIBUTE_PATH: path to the attributes extracted for the dataset. Attributes can be extracted using extract_attributes.py
        - NICO_DATA_FOLDER: path to the NICO dataset
    Only train_loader will return augmented samples
    The idx_train_loader will return original training samples along with their indexes in the training dataset

    Args:
        batch_size (int): batch size

    Returns:
        tuple[DataLoader]: train_loader, idx_train_loader, val_loader, test_loader
    """
    # get data transformations with (train_transform) and without (test_transform) data augmentations
    train_transform = get_transform_nico(train=True, augment_data=True)
    test_transform = get_transform_nico(train=False, augment_data=False)

    attribute_path = NICO_ATTRIBUTE_PATH

    trainset = NICO_dataset(
        basedir=NICO_DATA_FOLDER,
        split="train",
        transform=train_transform,
        attribute_embed=attribute_path,
    )
    trainset_ref = NICO_dataset(
        basedir=NICO_DATA_FOLDER,
        split="train",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    idx_train_loader = DataLoader(
        train_idx_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    valset = NICO_dataset(
        basedir=NICO_DATA_FOLDER,
        split="val",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    testset = NICO_dataset(
        basedir=NICO_DATA_FOLDER,
        split="test",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    return train_loader, idx_train_loader, val_loader, test_loader


def get_imagenet9_loader(batch_size: int):
    """Return train_loader, idx_train_loader, val_loader, and test_loader for the ImageNet-9/A datasets
    Note that train_loader, idx_train_loader, val_loader use ImageNet-9, and test_loader use ImageNet-A.

    Require the following constant specified in config.py:
        - IMAGENET9_ATTRIBUTE_PATH: path to the attributes extracted for the dataset. Attributes can be extracted using extract_attributes.py
        - IMAGENET9_DATA_FOLDER: path to the ImageNet-9 dataset
        - IMAGENETA_DATA_FOLDER: path to the ImageNet-A dataset
    Only train_loader will return augmented samples
    The idx_train_loader will return original training samples along with their indexes in the training dataset

    Args:
        batch_size (int): batch size

    Returns:
        tuple[DataLoader]: train_loader, idx_train_loader, val_loader, test_loader
    """

    # get data transformations with (train_transform) and without (test_transform) data augmentations
    train_transform = get_imagenet_transform(train=True, augment_data=True)
    test_transform = get_imagenet_transform(train=False, augment_data=False)

    attribute_path = IMAGENET9_ATTRIBUTE_PATH

    trainset = ImageNet9(
        basedir=IMAGENET9_DATA_FOLDER,
        split="train",
        transform=train_transform,
        attribute_embed=attribute_path,
    )
    trainset_ref = ImageNet9(
        basedir=IMAGENET9_DATA_FOLDER,
        split="train",
        transform=test_transform,
        attribute_embed=attribute_path,
    )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    idx_train_loader = DataLoader(
        train_idx_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    valset = ImageNet9(
        basedir=IMAGENET9_DATA_FOLDER,
        split="val",
        transform=test_transform,
        attribute_embed=attribute_path,
        cluster_file=IMAGENET9_VAL_CLUSTERS,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    testset = ImageNetA(
        basedir=IMAGENETA_DATA_FOLDER,
        transform=test_transform,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    return train_loader, idx_train_loader, val_loader, test_loader


def get_dataloader(dataset: str, batch_size: int, sampling: bool = True):
    """Get train_loader, idx_train_loader, val_loader, and test_loader based on the given dataset name as well as batch size

    Args:
        dataset (str): name of the dataset.
        batch_size (int): batch size
        sampling (bool, optional): set whether to sample a portion of training data to reduce training time. Default to True.

    Raises:
        ValueError: if the dataset name is not recognized

    Returns:
        tuple[DataLoader]: train_loader, idx_train_loader, val_loader, test_loader
    """
    if dataset == "waterbirds":
        loaders = get_waterbirds_loader(batch_size)
    elif dataset == "celeba":
        loaders = get_celeba_loader(batch_size, sampling)
    elif dataset == "nico":
        loaders = get_nico_loader(batch_size)
    elif dataset == "imagenet-9":
        loaders = get_imagenet9_loader(batch_size)
    else:
        raise ValueError("Unknow dataset")
    return loaders
