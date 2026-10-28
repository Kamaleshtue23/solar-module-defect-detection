#!/usr/bin/env python3
"""
Solar cell cropping from module images using a trained Mask R-CNN (ResNet-50).
"""
from PIL import Image
import cv2
import numpy as np
import os
import argparse
from mmcv import Config
from mmdet.apis import init_detector
from mmdet.apis.inference import inference_detector

# ---------- Helper functions ----------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def rotate_to_horizontal(image):
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def get_module_corners(processed_image, original_image):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    module_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(module_contour, True)
    approx = cv2.approxPolyDP(module_contour, epsilon, True)
    if len(approx) == 4:
        corners = np.array([point[0] for point in approx], dtype="float32")
    else:
        hull = cv2.convexHull(module_contour)
        corners = np.float32(cv2.boxPoints(cv2.minAreaRect(hull)))
    return order_points(corners)

# ---------- Model configuration ----------
img_norm_cfg = dict(mean=[123.875, 116.669, 103.939], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

cfg = dict(

    model=dict(

        type='MaskRCNN',

        pretrained=None,

        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            ),
            norm_eval=True,
            style='pytorch'
        ),

        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5
        ),

        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,

            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]
            ),

            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[1.0, 1.0, 1.0, 1.0]
            ),

            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0
            ),

            loss_bbox=dict(
                type='L1Loss',
                loss_weight=1.0
            )
        ),

        roi_head=dict(

            type='StandardRoIHead',

            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign',
                    output_size=7,
                    sampling_ratio=0
                ),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),

            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,

                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]
                ),

                reg_class_agnostic=False,

                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0
                ),

                loss_bbox=dict(
                    type='L1Loss',
                    loss_weight=1.0
                )
            ),

            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign',
                    output_size=14,
                    sampling_ratio=0
                ),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]
            ),

            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,

                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_mask=True,
                    loss_weight=1.0
                )
            )
        )
    ),

    test_cfg=dict(

        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(
                type='nms',
                iou_threshold=0.3
            ),
            min_bbox_size=0
        ),

        rcnn=dict(
            score_thr=0.3,
            nms=dict(
                type='nms',
                iou_threshold=0.5
            ),
            max_per_img=300,
            mask_thr_binary=0.5
        )
    ),

    data=dict(
        test=dict(
            type='CocoDataset',
            ann_file=None,
            img_prefix='',
            pipeline=test_pipeline
        )
    ),

    device='cuda'
)

from PIL import Image

def crop_solar_cells(model, image_path, confidence_threshold=0.5):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    img = rotate_to_horizontal(img)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((25, 25), np.uint8)

    dilated = cv2.dilate(
        otsu,
        kernel,
        iterations=1
    )

    closed = cv2.morphologyEx(
        dilated,
        cv2.MORPH_CLOSE,
        kernel
    )

    corners = get_module_corners(
        closed,
        img
    )

    if corners is None:

        print(
            "Warning: Could not detect module corners."
        )

        return []

    # Adjust corners

    corners[0] -= 5

    corners[1][0] += 5
    corners[1][1] -= 5

    corners[2] += 5

    corners[3][0] -= 5
    corners[3][1] += 5

    h, w = img.shape[:2]

    dst = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    M = cv2.getPerspectiveTransform(
        corners,
        dst
    )

    warped = cv2.warpPerspective(
        img,
        M,
        (w, h)
    )

    warped = gamma_correction(
        warped,
        gamma=1.5
    )

    # Run MaskRCNN

    result = inference_detector(
        model,
        warped
    )

    if isinstance(result, tuple):

        bboxes_per_class = result[0]

    else:

        bboxes_per_class = result

    bboxes = bboxes_per_class[0]

    if bboxes.ndim == 1:

        bboxes = bboxes.reshape(
            1,
            -1
        )

    detections = []

    for b in bboxes:

        conf = (
            b[4]
            if np.isscalar(b[4])
            else b[4].item()
        )

        if conf >= confidence_threshold:

            detections.append(
                b[:5]
            )

    if not detections:

        print(
            "No solar cells detected above confidence threshold."
        )

        return []

    widths = [
        b[2] - b[0]
        for b in detections
    ]

    heights = [
        b[3] - b[1]
        for b in detections
    ]

    mean_w = int(
        np.mean(widths)
    )

    mean_h = int(
        np.mean(heights)
    )

    cropped_cells = []

    for b in detections:

        x1, y1, x2, y2 = map(
            int,
            b[:4]
        )

        cell = warped[
            y1:y2,
            x1:x2
        ]

        cell = cv2.resize(
            cell,
            (mean_w, mean_h),
            interpolation=cv2.INTER_AREA
        )

        cell = apply_clahe(
            cell
        )

        # Convert OpenCV BGR -> RGB

        cell_rgb = cv2.cvtColor(
            cell,
            cv2.COLOR_BGR2RGB
        )

        # Convert to PIL for DeepLab

        cell_pil = Image.fromarray(
            cell_rgb
        )

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        cropped_cells.append({

            "image": cell_pil,

            "bbox": [
                x1,
                y1,
                x2,
                y2
            ],

            "center_x": cx,

            "center_y": cy

        })

    return cropped_cells

def process_folder(model, input_folder, output_folder, confidence=0.5):
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(image_extensions):
            in_path = os.path.join(input_folder, fname)
            print(f"Processing {fname}...")
            cells = crop_solar_cells(model, in_path, confidence_threshold=confidence)
            if cells:
                module_folder = os.path.join(output_folder, os.path.splitext(fname)[0])
                os.makedirs(module_folder, exist_ok=True)
                for i, cell in enumerate(cells):

                    out_path = os.path.join(
                        module_folder,
                        f"cell_{i+1}.jpg"
                    )

                    cell["image"].save(
                        out_path
                    )
                print(f"  Saved {len(cells)} cells to {module_folder}")
            else:
                print(f"  No cells found in {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop solar cells from module images")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or folder")
    parser.add_argument("--output", type=str, default="cropped_cells", help="Output folder")
    parser.add_argument("--model", type=str, default="latest (1).pth", help="Path to trained model checkpoint")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device: cuda:0 or cpu")
    args = parser.parse_args()

    cfg["device"] = args.device
    print(f"Loading model from {args.model}...")
    model = init_detector(Config(cfg), args.model, device=args.device)
    model.CLASSES = ('solar_cell',)

    if os.path.isdir(args.input):
        process_folder(model, args.input, args.output, args.conf)
    else:
        cells = crop_solar_cells(model, args.input, args.conf)
        if cells:
            os.makedirs(args.output, exist_ok=True)
            for i, cell in enumerate(cells):

                out_path = os.path.join(
                args.output,
                f"cell_{i+1}.jpg"
            )

                cell["image"].save(
                    out_path
                )
            print(f"Saved {len(cells)} cells to {args.output}")
        else:
            print("No cells detected.")