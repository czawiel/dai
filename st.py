import hashlib
import os
import streamlit as st
from st_audiorec import st_audiorec
import whisper

st.set_page_config(page_title="Dyktafon AI", page_icon="🎙️", layout="centered")

TEMP_AUDIO_PATH = "temp_recorded.wav"
OUTPUT_FILE = "transkrypcja.txt"


@st.cache_resource
def load_whisper_model(model_name: str = "base"):
    return whisper.load_model(model_name, device="cpu")


# Inicjalizacja stanu sesji
if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_audio_hash" not in st.session_state:
    st.session_state.last_processed_audio_hash = None

st.title("🎙️ Dyktafon AI")
st.caption("Nagrywaj z mikrofonu – po zatrzymaniu nagrania wynik pojawi się od razu automatycznie.")

# Panel boczny z ustawieniami
with st.sidebar:
    st.header("⚙️️ Ustawienia")
    selected_model = st.selectbox(
        "Model Whisper",
        options=["tiny", "base", "small"],
        index=1,
        help="Model 'base' oferuje świetny balans między szybkością a dokładnością na darmowych serwerach CPU.",
    )
    language_choice = st.selectbox(
        "Język transkrypcji",
        options=["pl", "en", "auto"],
        index=0,
        format_func=lambda x: "Polski" if x == "pl" else ("Angielski" if x == "en" else "Wykryj automatycznie"),
    )

with st.spinner(f"Ładowanie modelu '{selected_model}'..."):
    model = load_whisper_model(selected_model)

st.subheader("1. Nagraj dźwięk")
# Komponent nagrywający – po kliknięciu "Stop" natychmiast zwraca bajty audio
wav_audio_data = st_audiorec()

# Automatyczne przetwarzanie od razu po pojawieniu się nowego nagrania
if wav_audio_data is not None:
    # Obliczamy hash, aby transkrybować dane nagranie tylko jeden raz
    current_hash = hashlib.md5(wav_audio_data).hexdigest()

    if current_hash != st.session_state.last_processed_audio_hash:
        with st.spinner("⏳ Trwa automatyczna transkrypcja mowy na tekst..."):
            with open(TEMP_AUDIO_PATH, "wb") as f:
                f.write(wav_audio_data)

            try:
                lang = None if language_choice == "auto" else language_choice
                result = model.transcribe(
                    TEMP_AUDIO_PATH,
                    language=lang,
                    fp16=False,
                )
                recognized_text = result.get("text", "").strip()

                if recognized_text:
                    st.session_state.history.append(recognized_text)
                    st.toast("✅ Transkrypcja gotowa!", icon="🎉")
                else:
                    st.warning("Nie wykryto mowy w nagraniu.")

                st.session_state.last_processed_audio_hash = current_hash
            except Exception as exc:
                st.error(f"Wystąpił błąd transkrypcji: {exc}")
            finally:
                if os.path.exists(TEMP_AUDIO_PATH):
                    os.remove(TEMP_AUDIO_PATH)

st.divider()

# Wyświetlanie wyniku
st.subheader("2. Rozpoznany tekst")

full_transcription = "\n".join(st.session_state.history)

st.text_area(
    label="Wynik:",
    value=full_transcription,
    height=200,
    placeholder="Tu pojawi się przetłumaczony tekst...",
)

col_download, col_reset = st.columns([1, 1])

with col_download:
    if full_transcription:
        st.download_button(
            label="💾 Pobierz tekst (.txt)",
            data=full_transcription,
            file_name=OUTPUT_FILE,
            mime="text/plain",
            use_container_width=True,
        )

with col_reset:
    if st.button("🗑️ Wyczyść historię", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_processed_audio_hash = None
        st.rerun()
