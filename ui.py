import html as html_lib
import streamlit as st
import openai
from llm import humanize_text, get_available_models

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI → Human Text",
    page_icon="✍️",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0e0e10;
    color: #e8e6e0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #1a1020 0%, #0e0e10 60%);
}

h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f5c87a 0%, #e8a44a 50%, #c97b2e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
}

.subtitle {
    color: #6b6860;
    font-size: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.2rem;
    margin-bottom: 0.5rem;
}

.divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #e8a44a, transparent);
    margin: 0.4rem 0 1.2rem 0;
}

[data-testid="stTextArea"] textarea {
    background: #161618 !important;
    border: 1px solid #2a2820 !important;
    border-radius: 12px !important;
    color: #d4d0c8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #e8a44a !important;
    box-shadow: 0 0 0 3px rgba(232, 164, 74, 0.08) !important;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #e8a44a 0%, #c97b2e 100%) !important;
    color: #0e0e10 !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
}

[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

.col-label {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.col-label.ai       { color: #6688aa; }
.col-label.human    { color: #aa8844; }

.output-box {
    background: #111210;
    border: 1px solid #252520;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    min-height: 240px;
    font-size: 0.88rem;
    line-height: 1.8;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.output-box.human-box  { color: #d4b87a; border-color: #2a2415; }
.output-box.empty      { color: #3a3a38; font-style: italic; }

.badge {
    display: inline-block;
    background: #1a1a18;
    border: 1px solid #252520;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    color: #6b6860;
    margin-top: 0.5rem;
    margin-right: 0.4rem;
}
.badge span { color: #e8a44a; font-weight: 500; }

.pipeline-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #161410;
    border: 1px solid #2a2010;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    color: #8a7a5a;
    margin-bottom: 1rem;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def word_count(text): return len(text.split()) if text.strip() else 0
def badges(text):
    return f"""<div>
        <span class="badge">Words: <span>{word_count(text)}</span></span>
        <span class="badge">Chars: <span>{len(text)}</span></span>
    </div>"""


# ── Session state ─────────────────────────────────────────────────────────────
for key in ["humanized", "error"]:
    if key not in st.session_state:
        st.session_state[key] = ""


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>Humanize</h1>", unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI → Human pipeline · ESL Humanizer</p>', unsafe_allow_html=True)


# ── Settings ──────────────────────────────────────────────────────────────────
s1, s2, _ = st.columns([1.5, 1.5, 4])
with s1:
    selected_model = st.selectbox("Model", options=get_available_models(), index=0)
with s2:
    pass  # reserved for future settings

st.markdown("<br>", unsafe_allow_html=True)


# ── 2 Column layout ───────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<p class="col-label ai">⬡ AI Input</p>', unsafe_allow_html=True)
    ai_text = st.text_area(
        label="ai_input", label_visibility="collapsed",
        placeholder="Paste your AI-generated text here...",
        height=260, key="ai_input",
    )
    st.markdown(badges(ai_text), unsafe_allow_html=True)

with col2:
    st.markdown('<p class="col-label human">◈ Humanized</p>', unsafe_allow_html=True)
    if st.session_state.humanized:
        st.markdown(f'<div class="output-box human-box">{st.session_state.humanized}</div>', unsafe_allow_html=True)
        st.markdown(badges(st.session_state.humanized), unsafe_allow_html=True)
        dl_col, cp_col = st.columns(2)
        with dl_col:
            st.download_button("⬇ Download", data=st.session_state.humanized,
                               file_name="humanized.txt", mime="text/plain", key="dl_human")
        with cp_col:
            safe_text = html_lib.escape(st.session_state.humanized, quote=True)
            st.components.v1.html(f"""
                <button id="cb" data-text="{safe_text}"
                        style="background:linear-gradient(135deg,#e8a44a,#c97b2e);color:#0e0e10;border:none;border-radius:50px;
                               font-family:'DM Sans',sans-serif;font-weight:500;font-size:0.9rem;letter-spacing:0.05em;
                               padding:0.5rem 1.5rem;cursor:pointer;width:100%;">
                    ⎘ Copy
                </button>
                <script>
                document.getElementById('cb').addEventListener('click', function() {{
                    navigator.clipboard.writeText(this.dataset.text).then(() => {{
                        this.textContent = '✓ Copied';
                        setTimeout(() => this.textContent = '⎘ Copy', 1500);
                    }});
                }});
                </script>
            """, height=45)
    else:
        st.markdown('<div class="output-box empty">ESL humanized version appears here...</div>', unsafe_allow_html=True)


# ── Action buttons ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
b1, b2, _ = st.columns([1.3, 1, 5])

with b1:
    clicked = st.button("✦ Humanize", use_container_width=True)
with b2:
    if st.button("✕ Clear", use_container_width=True):
        st.session_state.humanized = ""
        st.session_state.error = ""
        st.rerun()

if st.session_state.error:
    st.error(st.session_state.error)


# ── Pipeline logic ────────────────────────────────────────────────────────────
if clicked:
    if not ai_text.strip():
        st.warning("Please paste some AI-generated text first.")
    else:
        st.session_state.error = ""
        with st.spinner("Humanizing..."):
            try:
                st.session_state.humanized = humanize_text(
                    ai_text, model=selected_model
                )
                st.rerun()
            except openai.AuthenticationError:
                st.session_state.error = "Invalid or expired OpenAI API key. Check your .env file."
                st.rerun()
            except openai.RateLimitError:
                st.session_state.error = "Rate limit hit. Please wait and try again."
                st.rerun()
            except openai.APIConnectionError:
                st.session_state.error = "Connection error — could not reach OpenAI. Check your internet connection and try again."
                st.rerun()
            except openai.APITimeoutError:
                st.session_state.error = "Request timed out. The text may be too long, or OpenAI is slow right now. Try again."
                st.rerun()
            except ValueError as e:
                st.session_state.error = str(e)
                st.rerun()
            except Exception as e:
                st.session_state.error = f"Unexpected error: {type(e).__name__}: {e}"
                st.rerun()