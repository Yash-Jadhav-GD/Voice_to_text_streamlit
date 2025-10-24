import os
import streamlit as st
import zipfile
import urllib.request
from vosk import Model

MODEL_DIR = "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

if not os.path.exists(MODEL_DIR):
    st.info("Downloading Vosk model (~50MB)...")
    zip_path = "model.zip"
    urllib.request.urlretrieve(MODEL_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)
    st.success("Vosk model downloaded and extracted!")

# Load the model
model = Model(MODEL_DIR)
st.success("Vosk model loaded!")
