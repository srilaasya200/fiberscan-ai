
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="FiberScan AI",
    page_icon="🧵",
    layout="wide"
)

MODEL_PATH = Path("fiberscan_mobilenetv3_finetuned_best.pth")

st.title("🧵 FiberScan AI")
st.subheader("AI-Powered Fabric Classification")
st.write(
    "Upload a fabric photo to estimate its likely category "
    "using our trained AI model."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        in_features, len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader(
    "Upload a fabric image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.image(
            image,
            caption="Uploaded fabric",
            use_container_width=True
        )

    try:
        model, class_names = load_model()
        tensor = preprocess(image).unsqueeze(0)

        with torch.inference_mode():
            probabilities = torch.softmax(
                model(tensor), dim=1
            )[0]

        top = torch.topk(
            probabilities, k=min(3, len(class_names))
        )

        best_index = top.indices[0].item()
        best_name = class_names[best_index]
        confidence = top.values[0].item() * 100

        with right:
            st.success("Prediction complete!")
            st.metric("Predicted category", best_name)
            st.metric("Model confidence", f"{confidence:.1f}%")

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee of correctness."
            )

            st.subheader("Other possible categories")
            for index, score in zip(
                top.indices[1:].tolist(),
                top.values[1:].tolist()
            ):
                st.write(
                    f"{class_names[index]} — {score * 100:.1f}%"
                )

    except Exception as error:
        st.error("The AI model could not run.")
        st.code(str(error))

st.caption(
    "Prototype only. Image-based predictions are estimates, "
    "not laboratory-confirmed fibre composition."
)
