### PartCrop: A Unified Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability

![PartCrop](https://github.com/JiePKU/MiniCrop/blob/master/img/PartCrop.JPG "PartCrop") 

### Requirements
* Python 3.8
* pytorch 1.12.0
* cuda 11.6
* timm 0.3.2
* numpy 1.22.3
* torchvision 0.13.0
* scikit-learn
* munkres
* itertools
* datetime
* pathlib
* PIL

### Quick Start

Self-supervised Pretraining with DINO on CIFAR100

```python
    python main_dino.py \
        --arch vit_small \
        --patch_size 8 \
        --img_size 32 \
        --crop_size 16 \
        --batch_size_per_gpu 1024 \
        --epochs 1600 \
        --data cifar100 \
        --output_dir /path/to/save/checkpoint \
        --num_workers 10 \
```

Train membership inference attacker using DINO on CIFAR100

```python
    python mia_test.py \
    --data cifar100 \
    --batch_size 100 \
    --model vit_small_patch8 \
    --img_size 32 \
    --model_path /pretrained/model/path/ \
    --epochs 100 \
    --lr 0.001 \
    --feature both \
    --global_crops_scale 0.2,1.0 \
    --local_crops_scale 0.08,0.2 \
    --local_crops_number 128 \
    --output_dir /path/to/save/attcker_checkpoint
```

Perform image classification on CIFAR100 using DINO

```python
    python eval_linear.py \
        --arch vit_small \
        --patch_size 8 \
        --img_size 32 \
        --pretrained_weights /path/to/pretrained_weights/ \
        --checkpoint_key student \
        --epochs 90 \
        --lr 0.001 \
        --batch_size_per_gpu 1024 \
        --data cifar100 \
        --num_workers 16 \
        --output_dir /path/to/save/linear_model \
        --num_labels 100 \
```
