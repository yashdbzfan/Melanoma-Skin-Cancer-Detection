import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Constants
MODEL_PATH = "melanoma_resnet50.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    
    # MATCHED ARCHITECTURE BASED ON YOUR PTH
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )

    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.error("Model loading failed due to mismatch!")
        st.text(str(e))
        raise

    model.eval()
    return model.to(DEVICE)

def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

st.title("Melanoma Skin Cancer Detection")
st.write("Upload a skin lesion image to predict if it is **benign** or **malignant**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        model = load_model()
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()

        if prob >= 0.5:
            st.error(f"Prediction: Malignant (Confidence: {prob:.2f})")
        else:
            st.success(f"Prediction: Benign (Confidence: {1 - prob:.2f})")
