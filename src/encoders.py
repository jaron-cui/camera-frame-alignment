import clip
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from transformers import ViTModel, ViTImageProcessor


def resnet_encoder(device) -> nn.Module:
    class CustomModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1).eval()
            self.layers = nn.Sequential(*list(model.children())[:-2])

        def forward(self, x):
            return self.layers(x).flatten(start_dim=1)

    return CustomModel().to(device)


def dino_hidden_state_encoder(device) -> nn.Module:
    class DinoWrapper(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.model = ViTModel.from_pretrained(model_name).to(device)
            # self.transform = T.Compose([
            #     T.Resize((224, 224)),
            #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])
            self.processor = ViTImageProcessor.from_pretrained(model_name)

        def forward(self, x):
            x = self.processor(images=x, return_tensors="pt").to(device)
            outputs = self.model(**x)
            return outputs.last_hidden_state[:, 1:, :].flatten(start_dim=1)

    return DinoWrapper('facebook/dino-vits16').to(device).eval()


def dino_cls_encoder(device) -> nn.Module:
    class DinoWrapper(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.model = ViTModel.from_pretrained(model_name).to(device)
            # self.transform = T.Compose([
            #     T.Resize((224, 224)),
            #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])
            self.processor = ViTImageProcessor.from_pretrained(model_name)

        def forward(self, x):
            x = self.processor(images=x, return_tensors="pt").to(device)
            outputs = self.model(**x)
            return outputs.last_hidden_state[:, 0, :]

    return DinoWrapper('facebook/dino-vits16').to(device).eval()


def clip_encoder(device) -> nn.Module:
    class ClipWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, _ = clip.load('ViT-B/32', device=device)
            self.preprocess = T.Compose([
                T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
                T.CenterCrop(size=(224, 224)),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
            # print(self.preprocess.transforms)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.preprocess(x)
            with torch.no_grad():
                image_features = self.model.encode_image(x)
            # print(image_features.shape)
            return image_features

    return ClipWrapper().to(device).eval()