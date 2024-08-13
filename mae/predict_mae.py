import cv2
from torchvision import transforms
import torch
from torch import nn
import numpy as np
import models_vit
from typing import List, Union

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


def normalize_image(image: np.ndarray, transform: transforms.Compose, image_size: tuple):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = transform(image)
    return image


def mae_predict(
        images: Union[List[np.array], np.array],
        model: nn.Module,
        transform: transforms.Compose,
        device: torch.device,
        image_size: tuple = (224, 224),
) -> np.array:
    """
    image: image in numpy array in bgr format or list of images in bgr format.
    """
    if not isinstance(images, list):
        images = [images]
    images = [normalize_image(image, transform, image_size) for image in images]
    images = torch.stack(images).to(device)

    prediction = model(images, use_head=False)
    prediction = prediction.detach().cpu().numpy()

    return prediction
