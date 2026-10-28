import streamlit as st


def show_cell_viewer(
    selected_cell
):

    st.markdown("---")

    st.subheader(
        f"Cell {selected_cell['cell_id']}"
    )

    col1, col2, col3 = st.columns(3)

    # ----------------------------------
    # Original
    # ----------------------------------

    with col1:

        st.image(
            selected_cell["image"],
            caption="Original Image",
            use_container_width=True
        )

    # ----------------------------------
    # Mask
    # ----------------------------------

    with col2:

        st.image(
            selected_cell["mask"],
            caption="Predicted Mask",
            use_container_width=True
        )

    # ----------------------------------
    # Overlay
    # ----------------------------------

    with col3:

        st.image(
            selected_cell["overlay"],
            caption="Overlay",
            use_container_width=True
        )

    st.markdown("### Detected Defects")

    defects = selected_cell["defects"]

    if len(defects) == 0:

        st.success(
            "No defects detected"
        )

        return

    for defect in defects:

        st.markdown(
            f"""
            <div style="
                padding:10px;
                margin-bottom:10px;
                border-radius:8px;
                background-color:#f3f6fb;
                border:1px solid #d9d9d9;
            ">

            <b>Defect Type:</b>
            {defect['name']}

            <br>

            <b>Confidence:</b>
            {defect['confidence']:.2f}

            <br>

            <b>Area Ratio:</b>
            {defect['area_ratio']:.4f}

            </div>
            """,
            unsafe_allow_html=True
        )