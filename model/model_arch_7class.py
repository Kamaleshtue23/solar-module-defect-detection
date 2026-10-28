import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Plus_ResNet7(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=3,
            classes=7
        )

    def forward(self, x):
        return self.model(x)