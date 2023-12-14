# Configurations for the Waterbirds dataset
WATERBIRDS_DATA_FOLDER = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2"
WATERBIRDS_CONCEPT_PATH_BLIP = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2/blip_img_embeddings_thre10_vocab160.pickle"
WATERBIRDS_VOCAB_PATH_BLIP = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2/blip_vocab_thre10_160.pickle"
WATERBIRDS_CONCEPT_PATH_VIT_GPT2 = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2/vit-gpt2_img_embeddings_thre10_vocab144.pickle"
WATERBIRDS_VOCAB_PATH_VIT_GPT2 = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2/vit-gpt2_vocab_thre10_144.pickle"
WATERBIRDS_ERM_MODEL = "/bigtemp/gz5hp/spurious_correlations/dfr_ckpts/waterbirds/erm_seed1/final_checkpoint.pt"

# Configurations for the CelebA dataset
CELEBA_DATA_FOLDER = "/bigtemp/gz5hp/dataset_hub/celebfaces/img_align_celeba/"
CELEBA_CONCEPT_PATH_VIT_GPT2 = "/bigtemp/gz5hp/dataset_hub/celebfaces/img_align_celeba/vit-gpt2_img_embeddings_thre10_vocab345.pickle"
CELEBA_VOCAB_PATH_VIT_GPT2 = "/bigtemp/gz5hp/dataset_hub/celebfaces/img_align_celeba/vocab_thre10_345.pickle"
CELEBA_ERM_MODEL = "/bigtemp/gz5hp/spurious_correlations/dfr_ckpts/celeba/erm_seed1/final_checkpoint.pt"
EXPR_PATH = "/bigtemp/gz5hp/unlearn_spurious_exprs/"