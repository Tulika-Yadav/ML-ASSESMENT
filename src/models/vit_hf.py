from pathlib import Path

import lightning as L
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]

_val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std),
    ]
)


class ViTLightningModule(L.LightningModule):
    def __init__(self):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=2,
            id2label={0: "Benign", 1: "Malignant"},
            label2id={"Benign": 0, "Malignant": 1},
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def pred_transforms(self, example_d):
        # print(example_d["image"])
        example_d["pixel_values"] = _val_transforms(example_d["image"].convert("RGB"))
        return example_d

    def predict(
        self,
        img_path: Path,
    ):
        image = Image.open(img_path)
        example_dict = {"image": image}
        inputs = self.pred_transforms(example_dict)
        inputs["pixel_values"] = inputs["pixel_values"].to(DEVICE)
        probab = self(inputs["pixel_values"].unsqueeze(0)).softmax(dim=1)
        label = self.vit.config.id2label[probab.argmax().item()]
        return label, probab
