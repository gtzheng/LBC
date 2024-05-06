# LBC
This is the code for the IJCAI 2024 Paper: *Learning Robust Classifiers with Self-Guided Spurious Correlation Mitigation*.
## Preparation

### Download datasets
- [Waterbirds](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz)
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) ([metadata](https://github.com/PolinaKirichenko/deep_feature_reweighting/blob/main/celeba_metadata.csv))
- [NICO](https://drive.google.com/drive/folders/17-jl0fF9BxZupG75BtpOqJaB6dJ2Pv8O?usp=sharing)
- ImageNet [(train](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) [,val)](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)
- [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)

Unzip the dataset files into individual folders.

In the `config.py` file, set `{dataset}_DATA_ROOT` to your corresponding dataset folder. 

### Prepare `metadata.csv` for each dataset
- Waterbirds and CelebA provide `metadata.csv` files.
- For the ImageNet-9 and ImageNet-A datasets, run the following code
    ```python
    from datasets.in9_data import prepare_imagenet9_metadata, prepare_imageneta_metadata
    base_dir = "path/to/imagenet/folder"
    prepare_imagenet9_metadata(base_dir)
    data_root = "path/to/imagenet-a/folder"
    prepare_imageneta_metadata(data_root)
    ````
- For the NICO dataset, run the following code
    ```python
    from datasets.nico_data import prepare_metadata
    prepare_metadata()
    ```

### Extracting attributes
For each dataset, run the following code:
```python
python extract_attributes.py --dataset waterbirds
python extract_attributes.py --dataset celeba
python extract_attributes.py --dataset nico
python extract_attributes.py --dataset imagenet-9
```

## Pretrain ERM models
For example, run the following code for the ImageNet-9 dataset
```python
python pretrain.py --dataset imagenet-9 --lr 0.001 --num_epochs 100 --pretrained_model
```
## LBC training
```python
python lbc_train.py --dataset imagenet-9 --lr 0.0001 --num_batches 100 --epoch 50 --K 4 --backbone resnet18
al
```
At the end of training, look for the corresponding results:
- Waterbirds, CelebA, and NICO: Find results starting with `[pseudo_val_unbiased]`.
- ImageNet-9: Find results starting with `[val_avg]`.


## Citation 
Please consider citing this paper if you find the code helpful.
```
@inproceedings{zheng2024learning,
 title={Learning Robust Classifiers with Self-Guided Spurious Correlation Mitigation},
 author={Zheng, Guangtao and Ye, Wenqian and Zhang, Aidong},
 booktitle={The 33rd International Joint Conference on Artificial Intelligence},
 year={2024}
}
```