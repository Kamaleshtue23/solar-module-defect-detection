import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from utils.mask_utils import (
    create_mask,
    decode_segmentation_mask
)


def predict_image(
    model,
    image,
    colormap,
    device
):

    # Convert NumPy -> PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Store original size for visualization
    orig_w, orig_h = image.size

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    img_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    model.eval()

    with torch.no_grad():
        pred = model(img_tensor)

    # Keep mask at model resolution (512x512)
    mask = create_mask(pred)

    color_mask = decode_segmentation_mask(
        mask,
        colormap
    )

    # Resize ONLY the visualization mask
    color_mask = np.array(
        Image.fromarray(color_mask).resize(
            (orig_w, orig_h),
            Image.NEAREST
        )
    )

    return pred, mask, color_mask