import hashlib
import os
import streamlit as st
import streamlit.components.v1 as components
from st_audiorec import st_audiorec
import whisper

st.set_page_config(page_title="Dyktafon AI", page_icon="🎙", layout="centered")

# Ukrycie podpowiedzi "Press Ctrl+Enter to apply" oraz wyśrodkowanie tytułu i napisów
st.markdown(
    """
    <style>
    div[data-testid="stTextAreaRootElement"] span {
        display: none !important;
    }
    .stTextArea [data-testid="InputInstructions"] {
        display: none !important;
    }
    .main-title, .main-caption {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ukrycie przycisków Reset i Download z wewnętrznej ramki iframe
components.html(
    """
    <script>
    const hideButtons = () => {
        const iframes = window.parent.document.querySelectorAll('iframe');
        iframes.forEach(iframe => {
            try {
                const doc = iframe.contentDocument || iframe.contentWindow.document;
                if (doc) {
                    const buttons = doc.querySelectorAll('button');
                    buttons.forEach(btn => {
                        const txt = btn.innerText.trim().toLowerCase();
                        if (txt === 'reset' || txt === 'download') {
                            btn.style.display = 'none';
                        }
                    });
                }
            } catch (e) {
                // ignoruj cross-origin
            }
        });
    };
    setInterval(hideButtons, 150);
    </script>
    """,
    height=0,
    width=0,
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
if "text_area_version" not in st.session_state:
    st.session_state.text_area_version = 0

# Wyśrodkowane logo na górze
if os.path.exists(LOGO_PATH):
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        st.image(LOGO_PATH)

st.markdown("<h1 class='main-title'>🎙️ Dyktafon AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='main-caption'>Kliknij 'Start Recording', aby nagrać mowę. Po kliknięciu 'Stop' tekst pojawi się automatycznie.</p>",
    unsafe_allow_html=True,
)

# Panel boczny z ustawieniami
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH)
    st.header("⚙ Ustawienia")
    selected_model = st.selectbox(
        "Model Whisper",
        options=["tiny", "base", "small"],
        index=1,
        help="Model 'base' oferuje świetny kompromis szybkości i dokładności na procesorach CPU.",
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

# Komponent nagrywający
wav_audio_data = st_audiorec()

# Automatyczna transkrypcja natychmiast po naciśnięciu "Stop"
if wav_audio_data is not None:
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

    # Przycisk pobierania pliku nagrania
    st.download_button(
        label="⬇️ Pobierz plik nagrania (.wav)",
        data=wav_audio_data,
        file_name="nagranie.wav",
        mime="audio/wav",
        use_container_width=True,
    )

st.divider()

# Wyświetlanie wyniku
st.subheader("2. Rozpoznany tekst")

full_transcription = "\n".join(st.session_state.history)

# Dynamiczny klucz gwarantuje natychmiastowe zresetowanie zawartości pola w widoku
st.text_area(
    label="Wynik:",
    value=full_transcription,
    height=200,
    placeholder="Tu pojawi się przetłumaczony tekst...",
    key=f"text_result_{st.session_state.text_area_version}",
)

col_download, col_reset = st.columns([1, 1])

with col_download:
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
        st.session_state.text_area_version += 1
        st.rerun()
