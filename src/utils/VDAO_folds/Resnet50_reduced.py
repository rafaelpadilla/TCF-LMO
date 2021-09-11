import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image


class Resnet50_Reduced():

    MEAN_IMAGENET = [0.485, 0.456, 0.406]
    STD_IMAGENET = [0.229, 0.224, 0.225]

    def __init__(self, device=None, pretrained=True, learning=False):
        self.device = device

        if device is None:
            device = 'cpu'

        # Corta a resnet50 para ficar somente com as camadas iniciais (até a residual3).
        self.resnet50 = torch.nn.Sequential(
            *list(models.resnet50(pretrained=pretrained).children())[0:5])
        if learning:
            self.resnet50.train()
        else:
            self.resnet50.eval()

        self.resnet50 = self.resnet50.to(self.device)

    def send_to_device(self, device):
        self.device = device
        self.resnet50 = self.resnet50.to(self.device)

    def freeze(self):
        for p in self.resnet50.parameters():
            p.requires_grad = False

    def __call__(self, frame_RGB):
        # image = Image.fromarray(frame_RGB.float().to(self.device))
        # t_img = self.transformations(image).float().to(self.device).unsqueeze(0)
        if frame_RGB.device == self.resnet50:
            return self.resnet50(frame_RGB)
        else:
            if frame_RGB.device.type != 'cuda':
                return self.resnet50(frame_RGB.to(self.device))
            else:
                return self.resnet50(frame_RGB)
