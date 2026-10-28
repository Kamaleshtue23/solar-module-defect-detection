import os

from PIL import Image


def load_cropped_cells(folder):

    cells = []

    files = sorted(
        [
            f
            for f in os.listdir(folder)
            if f.endswith(".jpg")
        ]
    )

    for file in files:

        row = int(
            file.split("_")[0][1:]
        ) - 1

        col = int(
            file.split("_")[1]
            .replace(".jpg", "")
            .replace("C", "")
        ) - 1

        image = Image.open(
            os.path.join(folder, file)
        ).convert("RGB")

        cells.append({

            "cell_id":
                file.replace(
                    ".jpg",
                    ""
                ),

            "row": row,

            "col": col,

            "image": image

        })

    return cells