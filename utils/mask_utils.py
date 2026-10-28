import numpy as np
import torch

def create_mask(pred_mask):

    pred_mask = torch.argmax(pred_mask, dim=1).squeeze(0)

    return pred_mask.cpu().numpy()


def decode_segmentation_mask(mask, colormap):

    h, w = mask.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in colormap.items():

        rgb[mask == label] = color

    return rgb