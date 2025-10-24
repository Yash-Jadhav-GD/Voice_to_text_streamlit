import streamlit as st
from vosk import Model, KaldiRecognizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import soundfile as sf
import audioread
import numpy as np
import io
import json

st.title("🎤 MP3/WAV Voice-to-Text + Summarization (Offline, CPU)")

st.markdown("""
Upload an MP3 or WAV file.  
Transcription uses Vosk, summarization uses Sumy (LSA).  
Fully offline, CPU-friendly, no PyTorch, no OpenAI, no pydub, no ffmpeg.
""")

audio_file = st.file_uploader("Upload MP3/WAV file", type=["mp3", "wav"])

# ✅ Update this to the absolute path where you extracted your Vosk model
VOSK_MODEL_PATH = r"C:\Users\yash.jadhav\Downloads\Voice_to_text\vosk-model-small-en-us-0.15"

# Function to read MP3/WAV using soundfile or audioread
def read_audio(file):
    try:
        # Try WAV first
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

    # Load Vosk model
    st.info("Loading Vosk model...")
    try:
        model = Model(VOSK_MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load Vosk model. Check path: {VOSK_MODEL_PATH}\n{e}")
        st.stop()

    rec = KaldiRecognizer(model, sr)

    # Transcription
    st.info("Transcribing audio...")
    text = ""
    step = 4000
    for i in range(0, len(y), step):
        chunk = y[i:i+step].tobytes()
        if rec.AcceptWaveform(chunk):
            result = rec.Result()
            res = json.loads(result)
            text += res.get("text", "") + " "

    st.subheader("📝 Transcribed Text")
    st.write(text)

    # Summarization
    if st.button("Summarize Text"):
        st.info("Summarizing...")
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count=3)
        summary = " ".join([str(s) for s in summary_sentences])
        st.subheader("📄 Summary")
        st.write(summary)

st.markdown("---")
st.caption("Offline MP3/WAV transcription + summarization using Vosk + Sumy (CPU only).")
