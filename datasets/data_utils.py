
from datasets.wb_data import (
    BiasedDataset,
    get_loader,
    get_transform_cub,
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
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


def get_waterbirds_loader(batch_size):
    train_transform = get_transform_cub(
        target_resolution=(224, 224), train=True, augment_data=True
    )
    test_transform = get_transform_cub(
        target_resolution=(224, 224), train=False, augment_data=False
    )
    
    attribute_path = WATERBIRDS_ATTRIBUTE_PATH_VIT_GPT2
    trainset = BiasedDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="train",
        transform=train_transform,
        attribute_embed=attribute_path
    )
    trainset_ref = BiasedDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="train",
        transform=test_transform,
        attribute_embed=attribute_path
    )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
    )
    idx_train_loader = DataLoader(
        train_idx_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    valset = BiasedDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="val",
        transform=test_transform,
        attribute_embed=attribute_path
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )

    testset = BiasedDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="test",
        transform=test_transform,
        attribute_embed=attribute_path
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    return train_loader, idx_train_loader, val_loader, test_loader


def get_celeba_loader(batch_size, sampling=True):
    train_transform = get_transform_cub(
        target_resolution=(224, 224), train=True, augment_data=True
    )
    test_transform = get_transform_cub(
        target_resolution=(224, 224), train=False, augment_data=False
    )
   
    attribute_path = CELEBA_ATTRIBUTE_PATH_VIT_GPT2
    
    trainset = BiasedDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="train",
        transform=train_transform,
        attribute_embed=attribute_path
    )
    trainset_ref = BiasedDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="train",
        transform=test_transform,
        attribute_embed=attribute_path
    )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
    )
    if sampling == True:
        idx_train_loader = DataLoader(
            train_idx_dataset,
            batch_size=batch_size,
            sampler=RandomSampler(
                train_idx_dataset, num_samples=batch_size*300),
            pin_memory=True,
            num_workers=4,
        )
    else:
        idx_train_loader = DataLoader(
            train_idx_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
        )
    valset = BiasedDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="val",
        transform=test_transform,
        attribute_embed=attribute_path
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )

    testset = BiasedDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="test",
        transform=test_transform,
        attribute_embed=attribute_path
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    return train_loader, idx_train_loader, val_loader, test_loader


def get_nico_loader(batch_size):
    train_transform = get_transform_nico(train=True, augment_data=True)
    test_transform = get_transform_nico(train=False, augment_data=False)
   
    attribute_path = NICO_ATTRIBUTE_PATH_VIT_GPT2

    trainset = NICO_dataset(
        basedir=NICO_DATA_FOLDER,
        split="train",
        training_dist=TRAINING_DIST,
        transform=train_transform,
        attribute_embed=attribute_path
    )
    trainset_ref = NICO_dataset(
        basedir=NICO_DATA_FOLDER,
        split="train",
        training_dist=TRAINING_DIST,
        transform=test_transform,
        attribute_embed=attribute_path
    )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
    )
    idx_train_loader = DataLoader(
        train_idx_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    valset = NICO_dataset(
        basedir=NICO_DATA_FOLDER,
        split="val",
        transform=test_transform,
        attribute_embed=attribute_path
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )

    testset = NICO_dataset(
        basedir=NICO_DATA_FOLDER,
        split="test",
        transform=test_transform,
        attribute_embed=attribute_path
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    return train_loader, idx_train_loader, val_loader, test_loader


def get_imagenet9_loader(batch_size):
    train_transform = get_imagenet_transform(train=True, augment_data=True)
    test_transform = get_imagenet_transform(train=False, augment_data=False)
   
    attribute_path = IMAGENET9_ATTRIBUTE_PATH_VIT_GPT2
   
    trainset = ImageNet9(
        basedir=IMAGENET9_DATA_FOLDER,
        split="train",
        transform=train_transform,
        attribute_embed=attribute_path
    )
    trainset_ref = ImageNet9(
        basedir=IMAGENET9_DATA_FOLDER,
        split="train",
        transform=test_transform,
        attribute_embed=attribute_path
    )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
    )
    idx_train_loader = DataLoader(
        train_idx_dataset,
        # sampler=RandomSampler(train_idx_dataset, num_samples=batch_size*2),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    valset = ImageNet9(
        basedir=IMAGENET9_DATA_FOLDER,
        split="val",
        transform=test_transform,
        attribute_embed=attribute_path,
        cluster_file=IMAGENET9_VAL_CLUSTERS
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )

    testset = ImageNetA(
        basedir=IMAGENETA_DATA_FOLDER,
        transform=test_transform,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
    )
    return train_loader, idx_train_loader, val_loader, test_loader


def get_dataloader(dataset, batch_size, sampling=True):
    if dataset == "waterbirds":
        loaders = get_waterbirds_loader(batch_size)
    elif dataset == "celeba":
        loaders = get_celeba_loader(batch_size, sampling)
    elif dataset == "nico":
        loaders = get_nico_loader(batch_size)
    elif dataset == "imagenet-9":
        loaders = get_imagenet9_loader(batch_size)
    else:
        raise ValueError(
            r"dataset must be from {waterbirds, celeba, imagenet-a, imagenet-9}")
    return loaders
