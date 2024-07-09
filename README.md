

## Environment
```
nvcr.io/nvidia/pytorch:23.11-py3
      +
timm==0.9.16
wandb
decord        # VideoMAE
```

# MAE

## Train
```bash
JOB_DIR="job_dirs"
DATA_DIR="path/to/dataset"

WANDB_API_KEY="..."
PROJECT_NAME="..."
EXPERIMENT_NAME="..."

python mae/main_pretrain.py \
    --output_dir ${JOB_DIR} \
    --data_path ${DATA_DIR} \
    --batch_size 32 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 50 \
    --warmup_epochs 5 \
    --num_workers 8 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --wandb_api_key ${WANDB_API_KEY} \
    --project ${PROJECT_NAME} \
    --name ${EXPERIMENT_NAME} 

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

## Predict (embeding)
```Python
import sys
import cv2
sys.path.append('mae')
import predict_mae

arch='vit_base_patch16'
checkpoint_path = "..."
image_path = "..."

model = predict_mae.create_mae_model(arch, checkpoint_path)

image = cv2.imread(image_path)
mae_embedding = predict_mae.mae_predcit(image, model, predict_mae.transform_mae)
```

# VideoMAE