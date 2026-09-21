from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import torch
import torch.nn as nn


CHECKPOINT_PATH = Path(__file__).with_name("simple_cnn_cats_dogs.pt")
DEFAULT_IMAGE_SIZE = 128
DEFAULT_CLASS_NAMES = ["Cat", "Dog"]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.name}. "
            "Run the export cell in the notebook first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = SimpleCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    return model, device, class_names, image_size


def prepare_image(image_bytes, image_size):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file could not be decoded as an image.") from error

    display_image = image.copy()
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).unsqueeze(0)
    return display_image, tensor


st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="C",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 850px; padding-top: 3rem; }
    .result { padding: 1rem 1.25rem; border-radius: 0.75rem; background: #eef6f1; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cat vs Dog Classifier")
st.caption("Upload one image and inspect the CNN prediction.")

try:
    model, device, class_names, image_size = load_model()
except (FileNotFoundError, KeyError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        display_image, image_tensor = prepare_image(uploaded_file.getvalue(), image_size)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.image(display_image, caption=uploaded_file.name, use_container_width=True)

    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor.to(device)), dim=1)[0].cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.markdown(
        f'<div class="result"><strong>Prediction: {predicted_class}</strong><br>'
        f'Confidence: {confidence:.1%}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Class probabilities")
    for class_name, probability in zip(class_names, probabilities.tolist()):
        st.write(f"{class_name}: {probability:.1%}")
        st.progress(probability)

st.caption(f"Running on {device}. Input resized to {image_size} x {image_size} pixels.")
