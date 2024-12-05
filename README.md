

## Environment
```
nvcr.io/nvidia/pytorch:23.11-py3
      +
timm==0.9.16
wandb
decord
```

# MAE
## Train
### Descriptions:
 - Train standard mae model
 - Training on multiple gpus

```bash
JOB_DIR="job_dirs"
DATA_DIR="data/cropped_clips"

WANDB_API_KEY="..."
PROJECT_NAME="..."
ENTITY_NAME="..."

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1  \
    mae/main_pretrain.py \
    --output_dir ${JOB_DIR} \
    --data_path ${DATA_DIR} \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 100 \
    --warmup_epochs 5 \
    --num_workers 8 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --wandb_api_key ${WANDB_API_KEY} \
    --project ${PROJECT_NAME} \
    --entity ${ENTITY_NAME}
```

## Predict
### Descriptions:
 - Predict online with pre-trained model

```Python
import sys
import cv2
import torch
sys.path.append('mae')
import predict_mae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arch='vit_base_patch16'
checkpoint_path = "..."
image_path = "..."

model = predict_mae.create_mae_model(arch, checkpoint_path)
model = model.to(device)

image = cv2.imread(image_path)
mae_embedding = predict_mae.mae_predcit(image, model, predict_mae.transform_mae, device)
```

## Create h5 features
### Descriptions:
 - Uses pre-trained mae model to predict the features on all clips and saves them as h5
 - Structure of the h5: `{"video_name_00": {clip_name_00: features_00_00, clip_name_01: features_00_01, ...}, ...}`
 - Shape of the features: `number of frames` x `embedding dimension`
 - Features are not normalized
 - VIT head is not used (`model_vit -> forward`)

```bash
python mae/create_mae_features.py \
  --input_folder data/cropped_clips \
  --output_folder data/features \
  --checkpoint checkpoints/model.pth \
  --arch vit_base_patch16 \
  --num_splits 10 \
  --split 0 \
  --dataset_name h2s \
  --split_name train \
  --annotation_file data\how2sign_realigned_train.csv   # only if the name is in bad format
```

