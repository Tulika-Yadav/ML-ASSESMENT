# ruff: noqa: F401
import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


compulsory_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def get_mob_grad_cam(model, img_path):
    model = model.to(DEVICE)
    target_layers = [model.model.features[18][0]]
    image = Image.open(img_path)

    image_used = np.array(image) / 255
    model_inp1 = torch.unsqueeze(
        compulsory_transforms(image=np.array(image))["image"], dim=0
    )
    input_tensor = model_inp1
    input_tensor = input_tensor.to(DEVICE)
    targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[targets[0]])
        grayscale_cam = grayscale_cam[0, :]
        visualization_ben = show_cam_on_image(
            image_used,
            grayscale_cam,
            use_rgb=True,
        )

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[targets[1]])
        grayscale_cam = grayscale_cam[0, :]
        visualization_mal = show_cam_on_image(
            image_used,
            grayscale_cam,
            use_rgb=True,
        )

    # w, h = visualization_ben.shape[1], visualization_ben.shape[0]
    # visualization_ben.resize((w // 2, h // 2, 3))
    # visualization_mal.resize((w // 2, h // 2, 3))
    # print(visualization_ben.size, visualization_mal.size)
    merged_img = np.hstack([visualization_ben, visualization_mal])
    # print(merged_img.shape)
    # return visualization_ben, visualization_mal
    return merged_img
