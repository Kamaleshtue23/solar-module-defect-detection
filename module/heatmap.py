import numpy as np
import plotly.graph_objects as go


def build_heatmap(results):

    if len(results) == 0:
        return go.Figure()

    max_rows = (
        max(
            r["row"]
            for r in results
        ) + 1
    )

    max_cols = (
        max(
            r["col"]
            for r in results
        ) + 1
    )

    z = np.full(
        (max_rows, max_cols),
        np.nan
    )

    text = [
        [""] * max_cols
        for _ in range(max_rows)
    ]

    for r in results:

        row = r["row"]
        col = r["col"]

        z[row, col] = r["severity"]

        text[row][col] = r["cell_id"]

    fig = go.Figure(

        data=go.Heatmap(

            z=z,

            text=text,

            texttemplate="%{text}",

            hovertemplate=(
                "Cell: %{text}<br>"
                "Severity: %{z}<extra></extra>"
            ),

            hoverongaps=False,

            colorscale=[
                [0.00, "#4CAF50"],
                [0.33, "#FFD54F"],
                [0.66, "#FB8C00"],
                [1.00, "#E53935"]
            ],

            zmin=0,
            zmax=1

        )
    )

    fig.update_layout(

        title="Module Defect Heatmap",

        xaxis_title="Column",

        yaxis_title="Row",

        height=700

    )

    fig.update_yaxes(
        autorange="reversed"
    )

    return fig