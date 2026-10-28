import numpy as np
from module.severity import get_severity
from inference.inference import predict_image
from inference.defect_detection import detect_defects


def run_module_pipeline(
        cropped_cells,
        model,
        class_info,
        colormap,
        thresholds,
        device,
        confidence_threshold,
        cell_threshold
):

    sorted_cells = cropped_cells

    results = []

    for cell in sorted_cells:

        image = cell["image"]

        logits, mask, color_mask = predict_image(
            model,
            image,
            colormap,
            device
        )

        defects = detect_defects(
            logits,
            mask,
            class_info,
            thresholds
        )

        defects = [
            d for d in defects
            if d["confidence"] >= confidence_threshold
        ]

        total_defect_area = sum(
            d["area_ratio"]
            for d in defects
        )

        max_confidence = max(
            [d["confidence"] for d in defects],
            default=0
        )

        defective = (
            max_confidence >= confidence_threshold
            and
            total_defect_area >= cell_threshold
        )

        severity = get_severity(
            total_defect_area
        )

        from PIL import Image
        import numpy as np

        original_np = np.array(image)

        h, w = original_np.shape[:2]

        color_mask_resized = np.array(
            Image.fromarray(color_mask).resize(
                (w, h)
            )
        )

        overlay = (
            0.6 * original_np +
            0.4 * color_mask_resized
        ).astype(np.uint8)

        results.append({

            "cell_id": cell["cell_id"],

            "row": cell["row"],
            "col": cell["col"],

            "image": image,

            "mask": color_mask,

            "overlay": overlay,

            "defects": defects,

            "severity": severity,

            "defective": defective,

            "total_defect_area": total_defect_area,

            "max_confidence": max_confidence

        })

    return results