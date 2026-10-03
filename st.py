import hashlib
import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from st_audiorec import st_audiorec
import whisper

st.set_page_config(page_title="Dyktafon AI", page_icon="🎙", layout="centered")

# Ukrycie podpowiedzi "Press Ctrl+Enter to apply" oraz wyśrodkowanie nagłówków
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

# Wstrzyknięcie JS: ukrycie Reset/Download oraz zmiana etykiety na "Zacznij Nagrywać"
components.html(
    """
    <script>
    const modifyButtons = () => {
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
                        } else if (txt === 'start recording') {
                            btn.innerText = 'Zacznij Nagrywać';
                        }
                    });
                }
            } catch (e) {
                // ignoruj cross-origin
            }
        });
    };

    setInterval(modifyButtons, 150);
    </script>
    """,
    height=0,
    width=0,
)

OUTPUT_FILE = "transkrypcja.txt"
LOGO_PATH = "logo.png"
README_PATH = "readme.pdf"


@st.cache_resource
def load_whisper_model(model_name: str = "base"):
    return whisper.load_model(model_name, device="cpu")


# Inicjalizacja stanu sesji
if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_audio_hash" not in st.session_state:
    st.session_state.last_processed_audio_hash = None

# Tytuł główny
st.markdown("<h1 class='main-title'>🎙️ Dyktafon AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='main-caption'>Kliknij 'Zacznij Nagrywać', powiedz coś i kliknij 'Stop'. Tekst pojawi się w polu poniżej.</p>",
    unsafe_allow_html=True,
)

# Panel boczny: Logo, Ustawienia na górze, Instrukcja na dole
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH)

    st.header("⚙ Ustawienia")
    selected_model = st.selectbox(
        "Model Whisper",
        options=["tiny", "base", "small"],
        index=1,
        help="Większy model = wyższa dokładność, ale dłuższy czas przetwarzania.",
    )
    language_choice = st.selectbox(
        "Język mowy",
        options=["pl", "en", "auto"],
        index=0,
        format_func=lambda x: "Polski"
        if x == "pl"
        else ("Angielski" if x == "en" else "Wykryj automatycznie"),
    )

    st.divider()

    st.header("📖 Instrukcja obsługi")
    st.markdown(
        """
        1. **Nagraj:** Kliknij `Zacznij Nagrywać` i mów do mikrofonu.
        2. **Zatrzymaj:** Kliknij `Stop` – transkrypcja rozpocznie się automatycznie.
        3. **Pobierz:** Zapisz gotowy plik `.txt` lub nagranie `.wav`.
        """
    )

    if os.path.exists(README_PATH):
        with open(README_PATH, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        st.download_button(
            label="📄 Pobierz pełną instrukcję (readme.pdf)",
            data=pdf_bytes,
            file_name="readme.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Plik readme.pdf będzie dostępny po umieszczeniu go w folderze.")

with st.spinner(f"Ładowanie modelu '{selected_model}'..."):
    model = load_whisper_model(selected_model)

st.subheader("1. Nagraj dźwięk")

# Komponent nagrywający z falą dźwiękową
wav_audio_data = st_audiorec()

# Automatyczna transkrypcja po zakończeniu nagrywania
if wav_audio_data is not None and len(wav_audio_data) > 0:
    current_hash = hashlib.md5(wav_audio_data).hexdigest()

    if current_hash != st.session_state.last_processed_audio_hash:
        with st.spinner("⏳ Rozpoznawanie mowy..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(wav_audio_data)
                tmp_path = tmp_file.name

            try:
                lang = None if language_choice == "auto" else language_choice
                
                result = model.transcribe(
                    tmp_path,
                    language=lang,
                    task="transcribe",
                    fp16=False,
                )
                recognized_text = result.get("text", "").strip()

                if recognized_text:
                    st.session_state.history.append(recognized_text)
                    st.toast("✅ Tekst został rozpoznany!", icon="🎉")
                else:
                    st.warning("Nagranie było za ciche lub nie rozpoznano słów.")

                st.session_state.last_processed_audio_hash = current_hash
            except Exception as exc:
                st.error(f"Błąd podczas transkrypcji: {exc}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # Przycisk pobierania pliku nagrania audio (.wav)
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

st.text_area(
    label="Wynik:",
    value=full_transcription,
    height=200,
    placeholder="Tu pojawi się rozpoznany tekst...",
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
    if st.button("🗑️ Wyczyść wynik", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_processed_audio_hash = None
        st.rerun()
