import streamlit as st
import os
import zipfile
import urllib.request
import soundfile as sf
import numpy as np
import io
import json
from vosk import Model, KaldiRecognizer
import streamlit.components.v1 as components

# Optional: only import moviepy when needed
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

st.title("🎤 Voice Transcription Tool 📝")
st.subheader("(Market Intelligence Team)")
st.header("Upload an MP3, WAV, or Video file (MP4, MOV, AVI)")

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
    st.success("Vosk model downloaded!")

# Load model
try:
    model = Model(MODEL_DIR)
except Exception as e:
    st.error(f"Failed to load Vosk model: {e}")
    st.stop()

# -------------------------
# 2️⃣ Audio/Video file uploader
# -------------------------
audio_file = st.file_uploader("Upload MP3, WAV, or Video", type=["mp3", "wav", "mp4", "mov", "avi"])

def read_audio(file, file_type):
    """Reads audio data from MP3, WAV, or video safely"""
    if file_type in ["mp4", "mov", "avi"]:
        if not MOVIEPY_AVAILABLE:
            st.error("moviepy is required to process video files. Install it using `pip install moviepy`.")
            return None, None
        
        st.info("Extracting audio from video...")
        temp_video_path = f"temp_video.{file_type}"
        with open(temp_video_path, "wb") as f:
            f.write(file.getbuffer())
        video = VideoFileClip(temp_video_path)
        
        if video.audio is None:
            st.error("Video file has no audio track.")
            video.close()
            os.remove(temp_video_path)
            return None, None
        
        temp_audio_path = "temp_audio.wav"
        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        data, sr = sf.read(temp_audio_path)
        video.close()
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
        return data, sr

    # Try WAV first
    try:
        data, sr = sf.read(file)
        return data, sr
    except:
        # Try MP3 using audioread
        import audioread
        file.seek(0)
        with audioread.audio_open(file) as f:
            sr = f.samplerate
            data = []
            for buf in f:
                samples = np.frombuffer(buf, dtype=np.int16)
                data.append(samples)
        data = np.concatenate(data)
        return data, sr

# -------------------------
# 3️⃣ Process file
# -------------------------
if audio_file is not None:
    file_type = audio_file.name.split(".")[-1].lower()
    
    # Play audio only for actual audio files
    if file_type in ["mp3", "wav"]:
        st.audio(audio_file, format="audio/wav")
    elif file_type in ["mp4", "mov", "avi"]:
        if MOVIEPY_AVAILABLE:
            st.info("Video uploaded. Audio will be extracted for transcription.")
        else:
            st.warning("Video uploaded but moviepy is not installed. Cannot process video.")
    
    y, sr = read_audio(audio_file, file_type)

    if y is None:
        st.stop()

    if y.dtype != np.int16:
        y = (y * 32767).astype(np.int16)
    st.success("File loaded successfully ✅")

    # -------------------------
    # 4️⃣ Transcription
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
    st.text_area("Transcription", value=text, height=300, disabled=True, key="transcription_box")

    # -------------------------
    # 5️⃣ Copy button using JS
    # -------------------------
    copy_html = f"""
    <textarea id="transcription_box_hidden" style="display:none;">{text}</textarea>
    <button onclick="navigator.clipboard.writeText(document.getElementById('transcription_box_hidden').value)">
        📋 Copy Text
    </button>
    """
    components.html(copy_html)

st.markdown("---")
st.caption("Click the Copy button to copy the text.")
st.markdown("---")
st.caption("In case of errors, contact yash.jadhav@gep.com or via Teams.")
