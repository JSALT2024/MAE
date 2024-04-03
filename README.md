# MAE

## Environment
```
nvcr.io/nvidia/pytorch:23.11-py3
      +
timm==0.9.16
```

## Train
```bash
JOB_DIR="job_dirs/experiment_name"
DATA_DIR="path/to/dataset"

mkdir ${JOB_DIR}

python mae/main_pretrain.py \
    --output_dir ${JOB_DIR} \
    --log_dir ${JOB_DIR} \
    --data_path ${DATA_DIR} \
    --batch_size 32 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 50 \
    --warmup_epochs 5 \
    --num_workers 8 \
    --blr 1.5e-4 \
    --weight_decay 0.05
```
```
path/to/dataset
└── train
    ├── class_0
    │    ├── image_0.png
    │    ├── image_1.png
    │    └── image_n.png
    ├── class_1
    └── class_k
```