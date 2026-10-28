import os
import zipfile
import shutil
from PIL import Image


def create_results_zip(results, zip_path="inspection_results.zip"):

    base_dir = "temp_results"
    root_folder = "inspection_results"

    # ------------------------------------------------
    # CLEAN OLD RESULTS
    # ------------------------------------------------
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    os.makedirs(f"{base_dir}/defective", exist_ok=True)
    os.makedirs(f"{base_dir}/non_defective", exist_ok=True)

    # ------------------------------------------------
    # SAVE FILES
    # ------------------------------------------------

    for r in results:

        folder = "defective" if r["defective"] else "non_defective"

        img_path = f"{base_dir}/{folder}/{r['name']}_image.png"
        mask_path = f"{base_dir}/{folder}/{r['name']}_mask.png"
        txt_path = f"{base_dir}/{folder}/{r['name']}_prediction.txt"

        r["image"].save(img_path)
        Image.fromarray(r["mask"]).save(mask_path)

        with open(txt_path, "w") as f:

            if len(r["defects"]) == 0:
                f.write("No defects detected\n")
            else:
                for d in r["defects"]:
                    f.write(
                        f"{d['name']} | confidence: {d['confidence']:.3f}\n"
                    )

    # ------------------------------------------------
    # CREATE ZIP WITH FIXED STRUCTURE
    # ------------------------------------------------

    with zipfile.ZipFile(zip_path, "w") as zipf:

        # create root folder
        zipf.writestr(f"{root_folder}/", "")

        # create subfolders
        zipf.writestr(f"{root_folder}/defective/", "")
        zipf.writestr(f"{root_folder}/non_defective/", "")

        for root, dirs, files in os.walk(base_dir):

            for file in files:

                path = os.path.join(root, file)

                relative_path = os.path.relpath(path, base_dir)

                zipf.write(
                    path,
                    f"{root_folder}/{relative_path}"
                )

    return zip_path