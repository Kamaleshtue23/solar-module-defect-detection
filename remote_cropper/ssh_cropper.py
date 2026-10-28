import os
import shutil
import zipfile
import subprocess


# ==========================================
# SERVER CONFIGURATION
# ==========================================

SERVER_USER = "vip-lab"

SERVER_IP = "10.107.35.100"

SERVER_ROOT = "/home/vip-lab/cropper_server"

SSH_TARGET = f"{SERVER_USER}@{SERVER_IP}"


# ==========================================
# MAIN FUNCTION
# ==========================================

def run_remote_cropper(local_image_path):

    os.makedirs("temp", exist_ok=True)

    # --------------------------------------
    # Upload image
    # --------------------------------------

    remote_input = (
        f"{SERVER_ROOT}/input/module.jpg"
    )

    subprocess.run(
        [
            "scp",
            local_image_path,
            f"{SSH_TARGET}:{remote_input}"
        ],
        check=True
    )

    # --------------------------------------
    # Run cropper remotely
    # --------------------------------------

    command = f"""
    source ~/anaconda3/etc/profile.d/conda.sh

    conda activate solar_cell

    cd {SERVER_ROOT}

    python run_cropper.py \
    --image input/module.jpg \
    --output outputs/module_cells
    """

    subprocess.run(
        [
            "ssh",
            SSH_TARGET,
            command
        ],
        check=True
    )

    # --------------------------------------
    # Download ZIP
    # --------------------------------------

    local_zip = "temp/cells.zip"

    remote_zip = (
        f"{SERVER_ROOT}/outputs/module_cells.zip"
    )

    subprocess.run(
        [
            "scp",
            f"{SSH_TARGET}:{remote_zip}",
            local_zip
        ],
        check=True
    )

    # --------------------------------------
    # Extract ZIP
    # --------------------------------------

    extract_dir = "temp/extracted_cells"

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)

    os.makedirs(
        extract_dir,
        exist_ok=True
    )

    with zipfile.ZipFile(
        local_zip,
        "r"
    ) as z:

        z.extractall(extract_dir)

    return extract_dir