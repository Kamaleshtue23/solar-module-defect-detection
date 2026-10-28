import pandas as pd

def load_class_info(csv_path):

    df = pd.read_csv(csv_path, encoding="ISO-8859-1")

    class_info = {}
    colormap = {}

    for _, row in df.iterrows():

        label = int(row["Label"])

        class_info[label] = {
            "desc": row["Desc"],
            "type": row["Class"]
        }

        colormap[label] = (
            int(row["Red"]),
            int(row["Green"]),
            int(row["Blue"])
        )

    return class_info, colormap