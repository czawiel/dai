import hashlib
import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from st_audiorec import st_audiorec
import whisper

st.set_page_config(page_title="Dyktafon AI", page_icon="🎙", layout="centered")

# Ukrycie podpowiedzi "Press Ctrl+Enter to apply", wyśrodkowanie nagłówków oraz styl dla logo
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
    .sidebar-logo img {
        border-radius: 0px !important;
        width: 100%;
        display: block;
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
LOGO_PATH = "logo3.svg"
README_PATH = "readme.pdf"
SELECTED_MODEL = "tiny"


@st.cache_resource
def load_whisper_model(model_name: str = "tiny"):
    return whisper.load_model(model_name, device="cpu")


# Inicjalizacja stanu sesji
if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_audio_hash" not in st.session_state:
    st.session_state.last_processed_audio_hash = None

# Tytuł główny i zmieniony tekst informacyjny
st.markdown("<h1 class='main-title'>🎙 Dyktafon AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='main-caption'>Kliknij 'Zacznij Nagrywać', powiedz coś i kliknij 'Stop'. Tekst pojawi się w polu poniżej. Staraj się mówić do mikrofonu wolno i wyraźnie.</p>",
    unsafe_allow_html=True,
)

# Panel boczny: Logo z linkiem, Ustawienia, Instrukcja, Kontakt
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "r", encoding="utf-8") as svg_file:
            svg_content = svg_file.read()
        st.markdown(
            f"""
            <div class="sidebar-logo">
                <a href="http://fabryka.tech/" target="_blank" rel="noopener noreferrer">
                    {svg_content}
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.header("⚙ U
