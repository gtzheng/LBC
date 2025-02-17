import os

EXPR_PATH = "/root/lbc_spurious_exprs/"
NUM_WORKERS = 4 # number of parallel data loaders

# Configurations for the Waterbirds dataset
WATERBIRDS_DATA_ROOT = "/root/dataset_hub/waterbird_complete95_forest2water2"
WATERBIRDS_DATA_FOLDER = WATERBIRDS_DATA_ROOT
WATERBIRDS_ATTRIBUTE_PATH = os.path.join(WATERBIRDS_DATA_FOLDER, "vit-gpt2_img_embeddings_thre10_vocab144.pickle")
WATERBIRDS_VOCAB_PATH = os.path.join(WATERBIRDS_DATA_FOLDER, "vit-gpt2_vocab_thre10_144.pickle")
WATERBIRDS_ERM_MODEL = "/root/lbc_spurious_exprs/pretrain_waterbirds/final_checkpoint.pt"

# Configurations for the CelebA dataset
CELEBA_DATA_ROOT = "/root/dataset_hub/celeba/"
CELEBA_DATA_FOLDER = os.path.join(CELEBA_DATA_ROOT, "img_align_celeba")
CELEBA_ATTRIBUTE_PATH = os.path.join(CELEBA_DATA_FOLDER, "vit-gpt2_img_embeddings_thre10_vocab345.pickle")
CELEBA_VOCAB_PATH = os.path.join(CELEBA_DATA_FOLDER, "vit-gpt2_vocab_thre10_345.pickle")
CELEBA_ERM_MODEL = "/root/lbc_spurious_exprs/pretrain_celeba/final_checkpoint.pt"
NUM_BATCHES = 300 # number of batches to sample per epoch

# NICO dataset
NICO_DATA_ROOT = "/root/dataset_hub/NICO"
NICO_DATA_FOLDER = os.path.join(NICO_DATA_ROOT, "multi_classification")
NICO_CXT_DIC_PATH = os.path.join(NICO_DATA_ROOT, "Context_name2label.json")
NICO_CLASS_DIC_PATH = os.path.join(NICO_DATA_ROOT, "Animal_name2label.json")
NICO_ATTRIBUTE_PATH = os.path.join(NICO_DATA_FOLDER, "vit-gpt2_img_embeddings_thre10_vocab199.pickle")
NICO_VOCAB_PATH = os.path.join(NICO_DATA_FOLDER, "vit-gpt2_vocab_thre10_199.pickle")



# Configurations for the ImageNet-9 dataset
IMAGENET_DATA_ROOT = "/root/dataset_hub/imagenet"
IMAGENETA_DATA_ROOT = "/root/dataset_hub/imagenet-a"

IMAGENET9_VAL_CLUSTERS = "./datasets/9class_imagenet_val.csv"
IMAGENET9_DATA_FOLDER = IMAGENET_DATA_ROOT
IMAGENETA_DATA_FOLDER = IMAGENETA_DATA_ROOT
IMAGENET9_ATTRIBUTE_PATH = os.path.join(IMAGENET9_DATA_FOLDER, "vit-gpt2_img_embeddings_thre10_vocab442.pickle")
IMAGENET9_VOCAB_PATH = os.path.join(IMAGENET9_DATA_FOLDER, "vit-gpt2_vocab_thre10_442.pickle")
IMAGENET9_ERM_MODEL = "/root/lbc_spurious_exprs/pretrain_imagenet-9_resnet18_01222024-163819_realai02/best_model.pt"

