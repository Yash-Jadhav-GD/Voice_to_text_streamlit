import streamlit as st
import os
import zipfile
import urllib.request
import audioread
import soundfile as sf
import numpy as np
import io
import json
from vosk import Model, KaldiRecognizer

st.title("🎤 MP3/WAV Voice-to-Text (Offline)")

st.markdown("""
Upload an MP3 or WAV file.  
Transcription uses Vosk and works fully offline after the first model download.
""")

# -------------------------
# 1️⃣ Vosk model auto-download
# -------------------------
MODEL_DIR = "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

if not os.path.exists(MODEL_DIR):
    st.info("Downloading Vosk model (~50MB)...")
    zip_path = "vosk_model.zip"
    urllib.request.urlretrieve(MODEL_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)
    st.success("Vosk model downloaded and extracted!")

# Load model
try:
    model = Model(MODEL_DIR)
except Exception as e:
    st.error(f"Failed to load Vosk model: {e}")
    st.stop()

# -------------------------
# 2️⃣ Audio file uploader
# -------------------------
audio_file = st.file_uploader("Upload MP3/WAV file", type=["mp3", "wav"])

def read_audio(file):
    # Try WAV first
    try:
        data, sr = sf.read(file)
        return data, sr
    except:
        # Try MP3 using audioread
        file.seek(0)
        with audioread.audio_open(file) as f:
            sr = f.samplerate
            data = []
            for buf in f:
                samples = np.frombuffer(buf, dtype=np.int16)
                data.append(samples)
        data = np.concatenate(data)
        return data, sr

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    st.info("Loading audio...")
    audio_bytes = audio_file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = read_audio(audio_buffer)

    # Convert float32 to int16 if needed
    if y.dtype != np.int16:
        y = (y * 32767).astype(np.int16)

    # -------------------------
    # 3️⃣ Transcription with progress bar
    # -------------------------
    st.info("Transcribing audio...")
    rec = KaldiRecognizer(model, sr)
    text = ""
    step = 4000
    progress_bar = st.progress(0)
    total_steps = len(y) // step + 1

    for i in range(0, len(y), step):
        chunk = y[i:i+step].tobytes()
        if rec.AcceptWaveform(chunk):
            result = rec.Result()
            res = json.loads(result)
            text += res.get("text", "") + " "
        progress_bar.progress(min((i//step + 1)/total_steps, 1.0))

    st.subheader("📝 Transcribed Text")
    # Non-editable text area
    st.text_area("Transcription", value=text, height=300, disabled=True)

st.markdown("---")
st.caption("Offline MP3/WAV transcription using Vosk (CPU only).")
