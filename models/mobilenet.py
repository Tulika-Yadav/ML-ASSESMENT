from pathlib import Path
from typing import Optional

import albumentations as A
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from albumentations.pytorch import ToTensorV2
from PIL import Image

MODEL_BACKBONE_WTS = tv_models.MobileNet_V2_Weights.IMAGENET1K_V2

compulsory_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


class classifictionModel(L.LightningModule):
    def __init__(
        self,
        lr: float,
        monitor_metric: str,
        monitor_mode: str,  # min | max
        pos_weight: Optional[float] = None,  # num positive / num negatives
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["model"])
        # ------------- build feature extractor and model -------------
        num_target_classes = 2
        model = tv_models.mobilenet_v2(weights=MODEL_BACKBONE_WTS)
        num_filters = model.classifier[1].in_features  # backbone.classifier.in_features
        model.classifier[1] = nn.Linear(num_filters, num_target_classes)
        self.model = model
        self.id2label = {0: "Benign", 1: "Malignant"}
        self.label2id = {"Benign": 0, "Malignant": 1}
        # -------------------------------------------------------------

    def forward(self, batch: torch.FloatTensor, probab=False):
        out = self.model(batch)
        if probab:
            out = out.softmax(dim=1)
        return out

    def predict_step(self, batch):
        return self(batch, probab=True)

    def predict(
        self,
        img_path: Path,
    ):
        img = Image.open(img_path)
        img = torch.unsqueeze(compulsory_transforms(image=np.array(img))["image"], dim=0)
        probab = self(img, probab=True)
        label = self.id2label[probab.argmax().item()]
        return label, probab
