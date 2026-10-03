import os
import streamlit as st
from st_audiorec import st_audiorec
import whisper

st.set_page_config(page_title="Dyktafon AI", page_icon="🎙️", layout="centered")

TEMP_AUDIO_PATH = "temp_recorded.wav"
OUTPUT_FILE = "transkrypcja.txt"


@st.cache_resource
def load_whisper_model(model_name: str = "base"):
    """
    Pobiera i buforuje model Whisper.
    Model 'base' lub 'small' jest zalecany dla darmowych serwerów chmurowych z CPU.
    """
    return whisper.load_model(model_name, device="cpu")


# Inicjalizacja historii transkrypcji w sesji
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🎙️ Dyktafon AI")
st.caption(
    "Nagrywaj bezpośrednio z mikrofonu w przeglądarce i transkrybuj za pomocą Whisper AI."
)

# Wybór modelu w panelu bocznym
with st.sidebar:
    st.header("⚙️ Ustawienia")
    selected_model = st.selectbox(
        "Model Whisper",
        options=["tiny", "base", "small"],
        index=1,
        help="Mniejsze modele ('tiny', 'base') działają szybciej na serwerach CPU.",
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
# Komponent nagrywający audio z poziomu przeglądarki użytkownika
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.subheader("2. Odtwarzacz")
    st.audio(wav_audio_data, format="audio/wav")

    col_transcribe, col_clear = st.columns([2, 1])

    with col_transcribe:
        start_transcription = st.button(
            "🚀 Transkrybuj nagranie", type="primary", use_container_width=True
        )

    if start_transcription:
        # Zapis tymczasowy pliku audio do przetworzenia przez Whisper
        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(wav_audio_data)

        with st.spinner("⏳ Trwa transkrypcja mowy na tekst..."):
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
                    st.success("Transkrypcja zakończona sukcesem!")
                else:
                    st.warning("Nie wykryto żadnej mowy w nagraniu.")

            except Exception as exc:
                st.error(f"Wystąpił błąd podczas transkrypcji: {exc}")
            finally:
                if os.path.exists(TEMP_AUDIO_PATH):
                    os.remove(TEMP_AUDIO_PATH)

st.divider()

# Wyświetlanie zebranego tekstu
st.subheader("3. Rozpoznany tekst")

full_transcription = "\n".join(st.session_state.history)

text_area = st.text_area(
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
        st.rerun()
