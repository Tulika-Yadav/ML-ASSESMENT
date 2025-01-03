# ruff: noqa: F401

import io
import shutil as sh
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import torch

from models.mobilenet import classifictionModel as MobileNetClf
from models.vit_hf import ViTLightningModule
from utils.mob_gradcam import get_mob_grad_cam
from utils.vit_gradcam import get_vit_grad_cam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_MODEL_PATH = r".\saved_models\vit-hf-epoch=08-valid_acc=0.0000.ckpt"
MOB_MODEL_PATH = r".\saved_models\mobile_net-epoch=15-valid_acc=0.8317.ckpt"
DEMO_IMG_PATH = r".\img_examples\test_mal_1091.jpg"

st.set_page_config(
    page_title="Skin cancer detection",
    page_icon="🫱🏽‍🫲🏼",
    # layout = 'wide'
)


@st.cache_resource
def load_vit_model(model_path: str | Path):
    model = ViTLightningModule.load_from_checkpoint(
        model_path,
        map_location=torch.device(DEVICE),
    )
    model.eval()
    # model.freeze()
    return model


@st.cache_resource
def load_mobilenet_model(model_path: str | Path):
    model = MobileNetClf.load_from_checkpoint(
        model_path,
        map_location=torch.device(DEVICE),
    )
    model.eval()
    return model


def plot_bar(ben_prob, mal_prob):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Benign", "Malignant"],
            x=[ben_prob, mal_prob],
            name="Probabilities",
            orientation="h",
            marker=dict(
                color="rgba(246, 78, 139, 0.6)",
                line=dict(color="rgba(246, 78, 139, 1.0)", width=1),
            ),
        )
    )
    return fig


def main():
    st.sidebar.title("Settings")
    st.sidebar.markdown("---")

    # check if GPU is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        st.sidebar.info("GPU available 🔥 - Predictions will be sped up")
    else:
        st.sidebar.warning("GPU NOT available 🚨 - Predictions might be slow")

    st.title("Skin Cancer Classification API")

    model_type = st.sidebar.selectbox(
        "Which model to use for classification?",
        options=["MobileNet", "ViT"],
        index=1,
    )

    st.sidebar.markdown("---")

    # upload image
    img_file_buffer = st.file_uploader("Upload a Valid Image", type=["jpg", "jpeg"])
    img_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    print(img_file.name)
    if img_file_buffer:
        file_bytes = io.BytesIO(img_file_buffer.read())
        with open(img_file.name, "wb") as f:
            f.write(file_bytes.read())
    else:
        sh.copy(DEMO_IMG_PATH, img_file.name)

    demo_bytes = img_file.read()
    st.markdown("## Input Image")

    st.sidebar.text("Original Image")
    st.sidebar.image(demo_bytes)

    match model_type:
        case "ViT":
            model = load_vit_model(VIT_MODEL_PATH)
        case "MobileNet":
            model = load_mobilenet_model(MOB_MODEL_PATH)
        case _:
            st.error("Invalid model type selected!")

    detection_button = st.button("Classify Image!")
    metric_placeholder = st.empty()

    if detection_button:
        label, probabs = model.predict(img_file.name)
        st.write(f"Predicted Label: {label}")
        probabs = probabs.detach().squeeze().numpy()
        ben_prob, mal_prob = probabs.tolist()
        ben_prob, mal_prob = round(ben_prob, 2), round(mal_prob, 2)

        # st.plotly_chart(plot_bar(0.5, 0.5))

        with metric_placeholder.container():
            col1, col2 = st.columns(2)
            col1.metric("Benign Probab", ben_prob)
            col2.metric("Malacious Probab", mal_prob)

        match model_type:
            case "ViT":
                vit_cam_img = get_vit_grad_cam(model, img_file.name)
                cam_img = vit_cam_img
                print(vit_cam_img.size)
            case "MobileNet":
                mob_cam_img = get_mob_grad_cam(model, img_file.name)
                cam_img = mob_cam_img
                # print(grad_ben.shape, grad_mal.shape)
            case _:
                st.error("Invalid model type selected!")

        st.image(
            cam_img,
            caption="Grad-CAM: Left - feature map of Benign \nRight - feature map of Malacious",
            use_container_width=True,
        )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # if not Path(MODEL_PATH).exists():
    #     drive_model_id = "1oQbFgQAxUM13m1nMmZ40QKNWho4Vllqq"
    #     g.download(id=drive_model_id, output=MODEL_PATH)

    main()
