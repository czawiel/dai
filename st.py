import hashlib
import os
from audio_recorder_streamlit import audio_recorder
import streamlit as st
import whisper

st.set_page_config(page_title="Dyktafon AI", page_icon="🎙️", layout="centered")

# Ukrycie podpowiedzi "Press Ctrl+Enter to apply" pod polem tekstowym
st.markdown(
    """
    <style>
    div[data-testid="stTextAreaRootElement"] span {
        display: none !important;
    }
    .stTextArea [data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TEMP_AUDIO_PATH = "temp_recorded.wav"
OUTPUT_FILE = "transkrypcja.txt"
LOGO_PATH = "logo.png"


@st.cache_resource
def load_whisper_model(model_name: str = "base"):
    return whisper.load_model(model_name, device="cpu")


# Inicjalizacja stanu sesji
if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_audio_hash" not in st.session_state:
    st.session_state.last_processed_audio_hash = None
if "current_audio" not in st.session_state:
    st.session_state.current_audio = None

# Logo i nagłówek
if os.path.exists(LOGO_PATH):
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image(LOGO_PATH, width=110)
    with col_title:
        st.title("🎙️ Dyktafon AI")
else:
    st.title("🎙️️ Dyktafon AI")

st.caption(
    "Kliknij przycisk, aby nagrać. Kliknij ponownie, aby zatrzymać – wynik pojawi się automatycznie."
)

# Panel boczny z ustawieniami
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)
    st.header("⚙ Ustawienia")
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
        format_func=lambda x: "Polski"
        if x == "pl"
        else ("Angielski" if x == "en" else "Wykryj automatycznie"),
    )

with st.spinner(f"Ładowanie modelu '{selected_model}'..."):
    model = load_whisper_model(selected_model)

st.subheader("1. Nagraj dźwięk")

# Komponent do nagrywania jednym kliknięciem
wav_audio_data = audio_recorder(
    text="Kliknij, aby nagrać",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    icon_name="microphone",
    icon_size="2x",
)

# Po zakończeniu nagrywania zapisujemy audio w stanie sesji
if wav_audio_data is not None:
    st.session_state.current_audio = wav_audio_data
    current_hash = hashlib.md5(wav_audio_data).hexdigest()

    # Automatyczna transkrypcja tylko dla nowego nagrania
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

# Wyświetlanie paska odtwarzacza i przycisku Download audio bezpośrednio pod nim
if st.session_state.current_audio is not None:
    st.audio(st.session_state.current_audio, format="audio/wav")
    st.download_button(
        label="⬇️ Pobierz nagranie audio (.wav)",
        data=st.session_state.current_audio,
        file_name="nagranie.wav",
        mime="audio/wav",
        use_container_width=True,
    )

st.divider()

# Wyświetlanie rozpoznanego tekstu
st.subheader("2. Rozpoznany tekst")

full_transcription = "\n".join(st.session_state.history)

st.text_area(
    label="Wynik:",
    value=full_transcription,
    height=200,
    placeholder="Tu pojawi się przetłumaczony tekst...",
)

col_download_txt, col_reset = st.columns([1, 1])

with col_download_txt:
    if full_transcription:
        st.download_button(
            label="💾 Pobierz transkrypcję (.txt)",
            data=full_transcription,
            file_name=OUTPUT_FILE,
            mime="text/plain",
            use_container_width=True,
        )

with col_reset:
    if st.button("🗑️ Wyczyść historię", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_processed_audio_hash = None
        st.session_state.current_audio = None
        st.rerun()
