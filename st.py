import os
import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
import streamlit as st
import whisper

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
SILENCE_THRESHOLD = 0.015
SILENCE_LIMIT_SEC = 1.3
PRE_PAD_CHUNKS = 4
OUTPUT_FILE = "transkrypcja.txt"

st.set_page_config(page_title="Dyktafon AI", page_icon="🎙️", layout="centered")


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small", device="cpu")


def transcribe(model, audio_data):
    """Zapisuje fragment do pliku tymczasowego i transkrybuje model Whisper."""
    temp_file = "temp_sentence.wav"
    try:
        audio_int16 = (audio_data * 32767).astype(np.int16)
        write(temp_file, SAMPLE_RATE, audio_int16)

        result = model.transcribe(
            temp_file,
            language="pl",
            fp16=False,
            beam_size=5,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
        )
        text = result.get("text", "").strip()
        text = text.replace("...", "").strip()
        return text
    except Exception as e:
        st.error(f"Błąd transkrypcji: {e}")
        return ""
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# Inicjalizacja stanu sesji
if "recording" not in st.session_state:
    st.session_state.recording = False
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []

st.title("🎙️ Dyktafon AI (Streamlit)")

with st.spinner("Ładowanie modelu Whisper..."):
    model = load_whisper_model()

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if not st.session_state.recording:
        if st.button("▶️ Start nasłuchiwania", use_container_width=True):
            st.session_state.recording = True
            st.rerun()

with col2:
    if st.session_state.recording:
        if st.button("⏹️ Zatrzymaj", use_container_width=True):
            st.session_state.recording = False
            st.rerun()

with col3:
    if st.button("🗑️ Wyczyść", use_container_width=True):
        st.session_state.transcripts = []
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        st.rerun()

# Kontenery do dynamicznego odświeżania widoku
status_box = st.empty()
log_box = st.empty()

# Wyświetlanie dotychczasowej transkrypcji
full_text = " ".join(st.session_state.transcripts)
log_box.text_area("Rozpoznany tekst:", value=full_text, height=260)

if st.session_state.recording:
    status_box.info("🔴 Nasłuchuję... Mów do mikrofonu.")

    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    silence_limit_chunks = int(SILENCE_LIMIT_SEC / CHUNK_DURATION)

    ring_buffer = []
    sentence_chunks = []
    is_speaking = False
    silent_chunks = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    ) as stream:
        while st.session_state.recording:
            chunk, _ = stream.read(chunk_samples)
            volume = np.max(np.abs(chunk))

            ring_buffer.append(chunk)
            if len(ring_buffer) > PRE_PAD_CHUNKS:
                ring_buffer.pop(0)

            if volume > SILENCE_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    sentence_chunks.extend(ring_buffer)
                else:
                    sentence_chunks.append(chunk)
                silent_chunks = 0
            elif is_speaking:
                sentence_chunks.append(chunk)
                silent_chunks += 1

                if silent_chunks >= silence_limit_chunks:
                    audio_full = np.concatenate(sentence_chunks, axis=0)
                    sentence_chunks = []
                    is_speaking = False
                    silent_chunks = 0

                    if len(audio_full) > SAMPLE_RATE * 0.7:
                        status_box.warning("⏳ Przetwarzanie transkrypcji...")
                        recognized = transcribe(model, audio_full)

                        if recognized:
                            st.session_state.transcripts.append(recognized)
                            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                                f.write(recognized + "\n")
                            # Aktualizacja pola tekstowego na żywo
                            log_box.text_area(
                                "Rozpoznany tekst:",
                                value=" ".join(st.session_state.transcripts),
                                height=260,
                            )
                        status_box.info("🔴 Nasłuchuję... Mów do mikrofonu.")
else:
    status_box.write("⏸️ Nasłuchiwanie wyłączone.")

# Opcja pobrania pliku z tekstem
if st.session_state.transcripts:
    st.download_button(
        label="💾 Pobierz transkrypcję (.txt)",
        data=" ".join(st.session_state.transcripts),
        file_name="transkrypcja.txt",
        mime="text/plain",
    )