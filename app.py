import streamlit as st
from vosk import Model, KaldiRecognizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from pydub import AudioSegment
import soundfile as sf
import io
import json

st.title("🎤 MP3 Voice-to-Text + Summarization (CPU, Offline)")

st.markdown("Upload an MP3 or WAV file. Transcription uses Vosk. Summarization uses Sumy.")

audio_file = st.file_uploader("Upload MP3/WAV file", type=["mp3", "wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Convert MP3 to WAV if needed
    audio_bytes = audio_file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    # Load WAV for Vosk
    data, samplerate = sf.read(wav_io)

    # Load Vosk model (download from https://alphacephei.com/vosk/models)
    st.info("Loading Vosk model...")
    model = Model("vosk-model-small-en-us-0.15")  # update path to your model

    rec = KaldiRecognizer(model, samplerate)

    st.info("Transcribing audio...")
    text = ""
    step = 4000
    for i in range(0, len(data), step):
        chunk = data[i:i+step].tobytes()
        if rec.AcceptWaveform(chunk):
            result = rec.Result()
            res = json.loads(result)
            text += res.get("text", "") + " "

    st.subheader("📝 Transcribed Text")
    st.write(text)

    if st.button("Summarize Text"):
        st.info("Summarizing...")
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, sentences_count=3)
        summary = " ".join([str(s) for s in summary_sentences])
        st.subheader("📄 Summary")
        st.write(summary)

st.markdown("---")
st.caption("Offline, CPU-friendly MP3 transcription + summarization using Vosk + Sumy.")
