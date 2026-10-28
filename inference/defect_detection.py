import numpy as np
import torch
import torch.nn.functional as F


def detect_defects(logits, mask, class_info, thresholds):

    probs = F.softmax(logits, dim=1).cpu().numpy()

    mask = mask.astype(np.int32)

    h, w = mask.shape
    total_pixels = h * w

    detected = []

    for label, info in class_info.items():

        if info["type"].lower() != "defect":
            continue

        pixels = (mask == label)

        area = np.sum(pixels)

        if area == 0:
            continue

        ratio = area / total_pixels

        threshold = thresholds.get(label, 0.001)

        if ratio > threshold:

            # probability of those pixels
            class_probs = probs[0, label][pixels]

            confidence = float(class_probs.mean())

            detected.append({
                "name": info["desc"],
                "confidence": confidence,
                "area_ratio": ratio
            })

    return detected