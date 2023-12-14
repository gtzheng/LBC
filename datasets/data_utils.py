
from datasets.wb_data import (
    WaterBirdsDataset,
    get_loader,
    get_transform_cub,
)
from config import *
from torch.utils.data import Dataset, DataLoader

class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


    
def get_waterbirds_loader(batch_size, vlm):
    train_transform = get_transform_cub(
            target_resolution=(224, 224), train=True, augment_data=True
        )
    test_transform = get_transform_cub(
        target_resolution=(224, 224), train=False, augment_data=False
    )
    if vlm == "blip":
        concept_path = WATERBIRDS_CONCEPT_PATH_BLIP
        assert "blip" in concept_path, "not blip-generated embeddings"
    elif vlm == "vit-gpt2":
        concept_path = WATERBIRDS_CONCEPT_PATH_VIT_GPT2
        assert "vit-gpt2" in concept_path, "not vit-gpt2-generated embeddings"
    assert "waterbird" in concept_path, "not from the waterbird dataset"
    assert "waterbird" in WATERBIRDS_DATA_FOLDER, "WATERBIRDS_DATA_FOLDER is incorrect"
    trainset = WaterBirdsDataset(
            basedir=WATERBIRDS_DATA_FOLDER,
            split="train",
            transform=train_transform,
            concept_embed=concept_path
        )
    trainset_ref = WaterBirdsDataset(
            basedir=WATERBIRDS_DATA_FOLDER,
            split="train",
            transform=test_transform,
            concept_embed=concept_path
        )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
                    trainset,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=4,
                )
    idx_train_loader = DataLoader(
                    train_idx_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=4,
                )
    valset = WaterBirdsDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="val",
        transform=test_transform,
        concept_embed=concept_path
    )
    val_loader = DataLoader(
                    valset,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=4,
                )

    testset = WaterBirdsDataset(
        basedir=WATERBIRDS_DATA_FOLDER,
        split="test",
        transform=test_transform,
        concept_embed=concept_path
    )
    test_loader = DataLoader(
                    testset,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=4,
                )
    return train_loader, idx_train_loader, val_loader, test_loader 

def get_celeba_loader(batch_size, vlm):
    train_transform = get_transform_cub(
            target_resolution=(224, 224), train=True, augment_data=True
        )
    test_transform = get_transform_cub(
        target_resolution=(224, 224), train=False, augment_data=False
    )
    if vlm == "blip":
        concept_path = CELEBA_CONCEPT_PATH_BLIP
        assert "blip" in concept_path, "not blip-generated embeddings"
    elif vlm == "vit-gpt2":
        concept_path = CELEBA_CONCEPT_PATH_VIT_GPT2
        assert "vit-gpt2" in concept_path, "not vit-gpt2-generated embeddings"
    assert "celeba" in concept_path, "not from the celeba dataset"
    assert "celeba" in CELEBA_DATA_FOLDER, "CELEBA_DATA_FOLDER is incorrect"
    trainset = WaterBirdsDataset(
            basedir=CELEBA_DATA_FOLDER,
            split="train",
            transform=train_transform,
            concept_embed=concept_path
        )
    trainset_ref = WaterBirdsDataset(
            basedir=CELEBA_DATA_FOLDER,
            split="train",
            transform=test_transform,
            concept_embed=concept_path
        )
    train_idx_dataset = IdxDataset(trainset_ref)
    train_loader = DataLoader(
                    trainset,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=4,
                )
    idx_train_loader = DataLoader(
                    train_idx_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=4,
                )
    valset = WaterBirdsDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="val",
        transform=test_transform,
        concept_embed=concept_path
    )
    val_loader = DataLoader(
                    valset,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=4,
                )

    testset = WaterBirdsDataset(
        basedir=CELEBA_DATA_FOLDER,
        split="test",
        transform=test_transform,
        concept_embed=concept_path
    )
    test_loader = DataLoader(
                    testset,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=4,
                )
    return train_loader, idx_train_loader, val_loader, test_loader 

def get_dataloader(dataset, batch_size, vlm):
    if dataset == "waterbirds":
        loaders = get_waterbirds_loader(batch_size, vlm)
    elif dataset == "celeba":
        loaders = get_celeba_loader(batch_size, vlm)
    else:
        raise ValueError(r"dataset must be from {waterbirds, celeba, bar, imagenet-a, imagenet-9}")
    return loaders
