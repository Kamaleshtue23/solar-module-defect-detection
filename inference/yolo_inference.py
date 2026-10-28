import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# ------------------------------
# MODEL PATH (update this)
# ------------------------------
MODEL_PATH = "model_weights/best.pt"

# ------------------------------
# LOAD MODEL
# ------------------------------
model = YOLO(MODEL_PATH)

CLASS_NAMES = [
    'Crack',
    'Inactive Areas',
    'Shunt Error',
    'Gridline Defect'
]


# ======================================================
# DETECTION FUNCTION (same logic as your code)
# ======================================================

def detect_defects(image_path, conf_threshold=0.15):

    results = model.predict(image_path, conf=conf_threshold, verbose=False)

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().item()
        cls = int(box.cls[0].cpu().item())

        area = (x2 - x1) * (y2 - y1)

        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(conf, 4),
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "area": int(area)
        })

    return detections, results


# ======================================================
# VISUALIZATION FUNCTION (adapted for Streamlit)
# ======================================================

def visualize_detections(image_path, detections):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} ({det['confidence']:.2f})"

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return img


# ======================================================
# STREAMLIT HELPER FUNCTION (NEW)
# ======================================================

def run_yolo_on_pil(image, conf_threshold=0.15):
    """
    Wrapper to use PIL image in Streamlit
    """

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    detections, _ = detect_defects(image_path, conf_threshold)
    vis_img = visualize_detections(image_path, detections)

    return detections, vis_img