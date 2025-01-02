# ruff : noqa F401

import warnings

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

warnings.filterwarnings("ignore")
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]


""" Model wrapper to return a tensor"""


class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits


""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.

"""


def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]


""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.

"""


def run_grad_cam_on_image(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    targets_for_gradcam: List[Callable],
    reshape_transform: Optional[Callable],
    input_tensor: torch.nn.Module,
    input_image: Image,
    method: Callable = GradCAM,
):
    with method(
        model=HuggingfaceToTensorModelWrapper(model),
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    ) as cam:
        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor, targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(
                np.float32(input_image) / 255, grayscale_cam, use_rgb=True
            )
            # Make it weight less in the notebook:
            visualization = cv2.resize(
                visualization, (visualization.shape[1] // 2, visualization.shape[0] // 2)
            )
            results.append(visualization)
        return np.hstack(results)


def print_top_categories(model, img_tensor, top_k=5):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k:][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")


def reshape_transform_vit_huggingface(x):
    activations = x[:, 1:, :]
    # print(activations.shape)
    # activations = activations.view(activations.shape[0], 12, 12, activations.shape[2])
    activations = activations.view(activations.shape[0], 14, 14, activations.shape[2])
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations


def get_vit_grad_cam(model, img_path):
    sub_model = model.vit
    targets_for_gradcam = [
        ClassifierOutputTarget(category_name_to_index(sub_model, "Benign")),
        ClassifierOutputTarget(category_name_to_index(sub_model, "Malignant")),
    ]
    # print(model)
    target_layer_gradcam = sub_model.vit.encoder.layer[-2].output
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std),
        ]
    )
    image = Image.open(img_path)
    image_resized = image.resize((224, 224))
    tensor_resized = _val_transforms(image)  # .to("cuda")

    out_img = Image.fromarray(
        run_grad_cam_on_image(
            model=sub_model,
            target_layer=target_layer_gradcam,
            targets_for_gradcam=targets_for_gradcam,
            input_tensor=tensor_resized,
            input_image=image_resized,
            reshape_transform=reshape_transform_vit_huggingface,
        )
    )

    # display(out_img)
    return out_img
