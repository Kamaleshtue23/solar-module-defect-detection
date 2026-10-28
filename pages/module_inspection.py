import streamlit as st

from module.heatmap import build_heatmap
from module.module_summary import build_summary


def show_module_results(
        results,
        show_overlay=True
):

    st.subheader(
        "Module Defect Heatmap"
    )

    heatmap = build_heatmap(
        results
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    st.markdown("---")

    summary = build_summary(
        results
    )

    st.subheader(
        "Inspection Summary"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Cells",
        summary["total"]
    )

    c2.metric(
        "Defective Cells",
        summary["defective"]
    )

    c3.metric(
        "Healthy Cells",
        summary["healthy"]
    )

    st.markdown("### Defect Distribution")

    for k, v in summary[
        "defect_counts"
    ].items():

        st.write(
            f"{k}: {v}"
        )

    st.markdown("---")

    cell_ids = [
        r["cell_id"]
        for r in results
    ]

    selected_cell = st.selectbox(
        "Select Cell",
        cell_ids
    )

    selected = next(
        r
        for r in results
        if r["cell_id"] == selected_cell
    )

    st.markdown(
        f"## Cell {selected_cell}"
    )

    if show_overlay:

        col1, col2, col3 = st.columns(3)

        col1.image(
            selected["image"],
            caption="Original Cell"
        )

        col2.image(
            selected["mask"],
            caption="Predicted Mask"
        )

        col3.image(
            selected["overlay"],
            caption="Overlay"
        )

    else:

        col1, col2 = st.columns(2)

        col1.image(
            selected["image"],
            caption="Original Cell"
        )

        col2.image(
            selected["mask"],
            caption="Predicted Mask"
        )

    st.markdown("---")

    if selected["defects"]:

        for d in selected["defects"]:

            st.markdown(
                f"""
                <div style="
                    border:1px solid #cccccc;
                    border-radius:10px;
                    padding:12px;
                    margin-bottom:10px;
                    background:#f5f5f5;
                    color:black;
                ">

                <b>Defect Type:</b>
                {d['name']}<br>

                <b>Confidence:</b>
                {d['confidence']:.3f}<br>

                <b>Area Ratio:</b>
                {d['area_ratio']:.5f}

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.success(
            "No defects detected"
        )