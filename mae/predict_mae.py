import os
import cv2
from torchvision import transforms
import torch
from torch import nn
import numpy as np
import models_vit

transform_mae = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def create_mae_model(arch: str, checkpoint_path: str = ""):
    model = models_vit.__dict__[arch](
        global_pool=False
    )
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)

        # missing head and unexpected decoder keys are expected
        print("Load checkpoint message:", msg)
    
    return model.eval()

def mae_predcit(image: np.array, model: nn.Module, transform: transforms.Compose, image_size: tuple = (224,224)):
    """
    image: image in numpy array in bgr format
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = transform(image).unsqueeze(0)
    
    prediction = model(image)
    prediction = prediction.detach().squeeze(0)
    
    return prediction