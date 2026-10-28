import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import tempfile

from ultralytics import YOLO

# DeepLab 7-Class
from model.model_arch_7class import DeepLabV3Plus_ResNet7

from inference.inference import predict_image
from inference.defect_detection import detect_defects

from utils.class_loader import load_class_info
from utils.zip_utils import create_results_zip

from config.thresholds_7class import (
    thresholds,
    CELL_DEFECT_THRESHOLD
)

# Module Workflow
from module.module_pipeline import (
    run_module_pipeline
)

from pages.module_inspection import (
    show_module_results
)


# ------------------------------------------------
# Page Configuration
# ------------------------------------------------

st.set_page_config(
    page_title="Solar Cell Inspection",
    layout="wide"
)

# ------------------------------------------------
# MODEL SELECTION PAGE
# ------------------------------------------------

st.markdown("## Select Inspection Model")

model_choice = st.radio(
    "Choose model:",
    [
        "DeepLabV3+ Detection Model",
        "YOLO Detection Model",
        "Module Inspection Workflow"
    ]
)

st.markdown("---")


# ============================================================
# ================= SEGMENTATION MODEL ========================
# ============================================================

if model_choice == "DeepLabV3+ Detection Model":

    # ==================== YOUR ORIGINAL CODE ====================

    st.markdown("## DeepLabV3+ Defect Detection")
    st.markdown("""
    <style>

    .main-title{
        font-size:44px;
        font-weight:700;
        color:#FF7043;
        margin-bottom:10px;
    }

    .section-title{
        font-size:30px;
        font-weight:600;
        color:#1A1A1A;
        margin-top:35px;
        margin-bottom:10px;
    }

    .result-card{
        padding:14px;
        border-radius:10px;
        border:1px solid #d9d9d9;
        background-color:#f3f6fb;
        color:#111111;
        font-size:16px;
        margin-bottom:12px;
    }

    .defect-box{
        padding:14px;
        border-radius:10px;
        background-color:#ffe5e5;
        color:#8b0000;
        font-size:22px;
        font-weight:600;
        margin-bottom:10px;
    }

    .ok-box{
        padding:14px;
        border-radius:10px;
        background-color:#e8f5e9;
        color:#1b5e20;
        font-size:22px;
        font-weight:600;
        margin-bottom:10px;
    }

    .stButton button{
        background-color:#1F4E79;
        color:white;
        border-radius:8px;
        height:45px;
        width:230px;
        font-size:16px;
    }

    .stDownloadButton button{
        background-color:#2E7D32;
        color:white;
        border-radius:8px;
        height:45px;
        font-size:16px;
    }

    [data-testid="stMetricValue"]{
        font-size:28px;
    }

    </style>
    """, unsafe_allow_html=True)

    # st.markdown("""
    # <div class='main-title'>
    # DeepLabv3+ Defect Detection
    # </div>
    # """, unsafe_allow_html=True)

    # st.markdown("## DeepLabV3+ Defect Detection")

    # st.write(
    # "Upload electroluminescence images of photovoltaic cells to automatically detect defects using a trained DeepLabV3+ segmentation model."
    # )

    st.sidebar.title("Inspection Settings")

    cell_threshold = st.sidebar.slider(
        "Cell Defect Area Threshold",
        min_value=0.0001,
        max_value=0.05,
        value=CELL_DEFECT_THRESHOLD,
        step=0.0001,
    )

    confidence_threshold = st.sidebar.slider(
        "Minimum Defect Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    show_overlay = st.sidebar.checkbox(
        "Show Overlay Visualization",
        value=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CSV_PATH = "/Users/kamaleshbarman/Documents/Final_streamlit_app/data/class_info_class6_ulta1.csv"
    MODEL_PATH = "/Users/kamaleshbarman/Documents/Final_streamlit_app/model_weights/deeplab_7class.pth"

    class_info, colormap = load_class_info(CSV_PATH)

    @st.cache_resource
    def load_model():

        model = DeepLabV3Plus_ResNet7()

        checkpoint = torch.load(MODEL_PATH, map_location=device)

        new_state = {}

        for k, v in checkpoint.items():

            if k.startswith("module."):
                k = k[7:]

            if not k.startswith("model."):
                k = "model." + k

            new_state[k] = v

        model.load_state_dict(new_state, strict=False)

        model.to(device)
        model.eval()

        return model


    model = load_model()

    # st.markdown(
    # "<div style='font-size:32px;font-weight:700;'>Upload EL Images</div>",
    # unsafe_allow_html=True
    # )

    uploaded_files = st.file_uploader(
        "Upload photovoltaic EL images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(f"Total images uploaded: {len(uploaded_files)}")

        results = []

        if st.button("Run Batch Inspection"):

            progress = st.progress(0)

            for i, file in enumerate(uploaded_files):

                image = Image.open(file).convert("RGB")

                logits, mask, color_mask = predict_image(
                    model, image, colormap, device
                )

                defects = detect_defects(
                    logits, mask, class_info, thresholds
                )

                defects = [
                    d for d in defects if d["confidence"] > confidence_threshold
                ]

                total_defect_area = sum(
                    d["area_ratio"] for d in defects
                )

                defective = total_defect_area > cell_threshold

                st.markdown("---")

                st.markdown(
                    f"<h3 style='color:#0B3C5D'>{file.name}</h3>",
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)

                col1.image(image, caption="Input Image")
                col2.image(color_mask, caption="Predicted Mask")

                if show_overlay:
                    from PIL import Image

                    image_np = np.array(image)

                    h, w = image_np.shape[:2]

                    color_mask_resized = np.array(
                        Image.fromarray(color_mask).resize(
                            (w, h)
                        )
                    )

                    overlay = (
                        image_np * 0.6
                        +
                        color_mask_resized * 0.4
                    ).astype(np.uint8)

                if defective:
                    st.markdown("<div class='defect-box'>Defective Cell</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='ok-box'>Acceptable Cell</div>", unsafe_allow_html=True)

                # if defects:
                #     for d in defects:
                #         st.write(f"{d['name']} ({d['confidence']:.2f})")
                if defects:

                    for d in defects:

                        st.markdown(
                            f"""
                            <div class="result-card">
                            <b>Defect Type:</b> {d['name']} <br>
                            <b>Confidence:</b> {d['confidence']:.2f} <br>
                            <b>Area Ratio:</b> {d['area_ratio']:.4f}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                else:
                    st.write("No defects detected")


# ============================================================
# ======================= YOLO MODEL ==========================
# ============================================================

elif model_choice == "YOLO Detection Model":

    st.markdown("## YOLO Defect Detection")

    YOLO_MODEL_PATH = "/Users/kamaleshbarman/Documents/Final_streamlit_app/model_weights/best.pt"

    CLASS_NAMES = [
        'Crack',
        'Inactive Areas',
        'Shunt Error',
        'Gridline Defect'
    ]

    st.sidebar.title("Inspection Settings")

    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.01, 1.0, 0.15, 0.01
    )

    @st.cache_resource
    def load_yolo():
        return YOLO(YOLO_MODEL_PATH)

    model = load_yolo()

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("Run YOLO Detection"):

            for file in uploaded_files:

                image = Image.open(file).convert("RGB")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    path = tmp.name

                results = model.predict(path, conf=conf_threshold, verbose=False)

                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                detections = []

                for box in results[0].boxes:

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    cls = int(box.cls[0].cpu().item())

                    detections.append(CLASS_NAMES[cls])

                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(img, f"{CLASS_NAMES[cls]} {conf:.2f}",
                                (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                st.markdown(f"### {file.name}")
                st.image(img)

                if detections:
                    st.error("Defects Detected")
                else:
                    st.success("No defects detected")
# ============================================================
# ================ MODULE INSPECTION WORKFLOW =================
# ============================================================

else:

    st.markdown("## Module Inspection Workflow")

    st.sidebar.title("Inspection Settings")

    cell_threshold = st.sidebar.slider(
        "Cell Defect Area Threshold",
        min_value=0.0001,
        max_value=0.05,
        value=CELL_DEFECT_THRESHOLD,
        step=0.0001,
    )

    confidence_threshold = st.sidebar.slider(
        "Minimum Defect Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    show_overlay = st.sidebar.checkbox(
        "Show Overlay Visualization",
        value=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    CSV_PATH = "data/class_info_class6_ulta1.csv"

    MODEL_PATH = "model_weights/deeplab_7class.pth"

    class_info, colormap = load_class_info(
        CSV_PATH
    )

    @st.cache_resource
    def load_deeplab():

        model = DeepLabV3Plus_ResNet7()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device
        )

        new_state = {}

        for k, v in checkpoint.items():

            if k.startswith("module."):
                k = k[7:]

            if not k.startswith("model."):
                k = "model." + k

            new_state[k] = v

        model.load_state_dict(
            new_state,
            strict=False
        )

        model.to(device)

        model.eval()

        return model

    deeplab_model = load_deeplab()
    if "module_results" not in st.session_state:
        st.session_state.module_results = None

    if "module_processed" not in st.session_state:
        st.session_state.module_processed = False

    uploaded_module = st.file_uploader(
        "Upload Module EL Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_module:

        module_img = Image.open(
            uploaded_module
        ).convert("RGB")

        st.image(
            module_img,
            caption="Uploaded Module"
        )

        if st.button(
            "Run Module Inspection"
        ):

            with st.spinner(
                "Cropping Cells..."
            ):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as tmp:

                    module_img.save(tmp.name)

                    temp_image_path = tmp.name

                from remote_cropper.ssh_cropper import (
                    run_remote_cropper
                )

                from module.load_cells import (
                    load_cropped_cells
                )
                
                extracted_dir = run_remote_cropper(
                    temp_image_path
                )

                cropped_cells = load_cropped_cells(
                    extracted_dir
                )
                st.session_state.cropped_cells = cropped_cells

            st.success(
                f"{len(cropped_cells)} cells extracted"
            )

            with st.spinner(
                "Running DeepLab Inspection..."
            ):

                results = run_module_pipeline(
                    cropped_cells=cropped_cells,
                    model=deeplab_model,
                    class_info=class_info,
                    colormap=colormap,
                    thresholds=thresholds,
                    device=device,
                    confidence_threshold=confidence_threshold,
                    cell_threshold=cell_threshold
                )

                st.session_state.module_results = results
                st.session_state.module_processed = True
        if st.session_state.module_processed:

            show_module_results(
                st.session_state.module_results,
                show_overlay=show_overlay
            )