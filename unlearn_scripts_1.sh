#!/bin/sh
python unlearn_identified_features.py --topk 5 --alpha 0.2 --vlm blip --iter_num 50
python unlearn_identified_features.py --topk 5 --alpha 0.2 --vlm vit-gpt2 --iter_num 50 --tolerance 5 --resume /bigtemp/gz5hp/unlearn_spurious_exprs/waterbirds_vit-gpt2_Iter_10_K_5_B_128_lr_0.0001_lrd_0.0100_alpha_0.20_epoch_10_epochd_10_debug/last_model.pt