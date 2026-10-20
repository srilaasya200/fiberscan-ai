import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="FiberScan AI",
    page_icon="🧵",
    layout="wide"
)

st.title("🧵 FiberScan AI")
st.subheader("AI-Powered Fabric Classification")

st.write(
    "Upload a fabric photo to analyse its visual characteristics "
    "and classify its likely fabric category."
)

uploaded_file = st.file_uploader(
    "Upload a fabric image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded fabric", use_container_width=True)

    st.info(
        "Image received successfully! The AI model will be connected "
        "in the next step."
    )

st.caption(
    "Prototype only. Image-based predictions are estimates, "
    "not laboratory-confirmed fibre composition."
)
