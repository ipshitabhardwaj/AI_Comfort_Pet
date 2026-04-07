"""
Emotion-Aware Digital Comfort Pet 🌸

Author: Ipshita Bhardwaj
  - Breathing exercise widget (box / 4-7-8 / calm)
  - Journaling prompts per emotion
  - Coping strategies panel
  - Mood trend mini-chart
  - Streak tracking
  - Improved confidence display
  - Richer animations
  - Sarcasm detection indicator
  - Multi-emotion blend display
"""

import streamlit as st
import pickle
import json
from datetime import datetime, date
from pathlib import Path
import time
import base64

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None

from utils import (
    detect_emotion_with_confidence,
    get_comfort_response,
    get_emotion_color,
    get_pet_reaction,
    analyze_mood_trend,
    get_emotion_emoji,
    get_journal_prompt,
    get_coping_strategies,
    get_breathing_exercise,
    EMOTION_GRADIENTS,
)

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Comfort Pet",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================== STYLES ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,300;0,400;0,600;0,700;0,800;1,400&family=Playfair+Display:ital,wght@0,700;1,400&display=swap');

*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

html, body, .stApp {
    background: #fff5f8 !important;
    font-family: 'Nunito', sans-serif;
    color: #3d1a2e;
}

#MainMenu, footer, header, .stDeployButton { display:none !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1300px !important; margin: 0 auto; }

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 10% 10%, rgba(255,182,213,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 90%, rgba(255,220,235,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 0%, rgba(255,200,225,0.1) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
}

/* ── Typography ── */
.site-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.6rem, 5vw, 4rem);
    font-weight: 700;
    color: #e8427c;
    letter-spacing: -1px;
    line-height: 1.1;
}
.site-sub {
    font-size: 0.82rem;
    color: #c47a99;
    letter-spacing: 4px;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 0.3rem;
}
.bow-accent { font-size: 2rem; display: block; margin-bottom: 0.4rem; }

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.85);
    border: 2px solid #fce4ec;
    border-radius: 28px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(232,66,124,0.06);
}
.card-pet {
    background: linear-gradient(145deg, rgba(255,240,246,0.95), rgba(255,252,254,0.9));
    border: 2px solid #f8bbd0;
}
.card-breathe {
    background: linear-gradient(135deg, rgba(176,123,232,0.12), rgba(232,66,124,0.08));
    border: 2px solid #dcc8f8;
    border-radius: 24px;
    padding: 1.5rem;
}
.card-journal {
    background: linear-gradient(135deg, rgba(123,142,240,0.1), rgba(232,66,124,0.06));
    border: 2px solid #c8d4f8;
    border-radius: 24px;
    padding: 1.5rem;
}
.card-coping {
    background: linear-gradient(135deg, rgba(96,200,160,0.1), rgba(232,66,124,0.06));
    border: 2px solid #b0e8d0;
    border-radius: 24px;
    padding: 1.5rem;
}

/* ── Emotion pill ── */
.emotion-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 1.2rem;
    border-radius: 100px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    border: 2px solid;
}

/* ── Confidence bar ── */
.conf-wrap {
    background: #fce4ec;
    border-radius: 100px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin: 0.25rem 0 1.5rem;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1.2s cubic-bezier(.4,0,.2,1);
}

/* ── Response text ── */
.response-text {
    font-size: 1rem;
    line-height: 1.9;
    color: #5a2040;
    font-weight: 400;
}

/* ── Pet area ── */
.pet-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.85rem;
}
.pet-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #e8427c;
}
.pet-mood-text {
    font-size: 0.9rem;
    color: #c47a99;
    text-align: center;
    min-height: 1.4rem;
    letter-spacing: 0.3px;
    font-weight: 600;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: rgba(252,228,236,0.2) !important;
    border: 2px solid #f8bbd0 !important;
    border-radius: 18px !important;
    color: #3d1a2e !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem 1.2rem !important;
    resize: none !important;
    transition: border-color 0.3s !important;
}
.stTextArea textarea:focus {
    border-color: #e8427c !important;
    box-shadow: 0 0 0 3px rgba(232,66,124,0.1) !important;
}
.stTextArea textarea::placeholder { color: #d4a0b8 !important; }
.stTextArea label { display: none !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e8427c 0%, #f06aaa 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 18px rgba(232,66,124,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(232,66,124,0.4) !important;
}

/* ── Mood entries ── */
.mood-entry {
    background: rgba(255,255,255,0.8);
    border-radius: 16px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
    border-left: 4px solid;
    border-top: 1px solid #fce4ec;
    border-right: 1px solid #fce4ec;
    border-bottom: 1px solid #fce4ec;
    transition: background 0.2s;
}
.mood-entry:hover { background: rgba(252,228,236,0.4); }
.mood-entry-emotion { font-weight: 800; font-size: 0.8rem; letter-spacing: 1.5px; text-transform: uppercase; }
.mood-entry-text { font-size: 0.88rem; color: #a0527a; margin-top: 0.25rem; }
.mood-entry-time { font-size: 0.75rem; color: #c8a0b8; }

/* ── Stat cards ── */
.stat-card {
    background: rgba(255,255,255,0.85);
    border: 2px solid #fce4ec;
    border-radius: 18px;
    padding: 1.1rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(232,66,124,0.05);
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e8427c;
}
.stat-label { font-size: 0.72rem; color: #c47a99; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0.2rem; font-weight: 600; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent !important;
    border-bottom: 2px solid #fce4ec !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #c47a99 !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    border-radius: 10px 10px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(232,66,124,0.08) !important;
    color: #e8427c !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #fff0f5 !important;
    border-right: 2px solid #fce4ec !important;
}
section[data-testid="stSidebar"] * { color: #5a2040 !important; }

hr { border-color: #fce4ec !important; margin: 1.5rem 0 !important; border-width: 2px !important; }
[data-testid="stMetricValue"] { color: #e8427c !important; }
.stAlert {
    background: rgba(252,228,236,0.5) !important;
    border: 2px solid #f8bbd0 !important;
    border-radius: 14px !important;
    color: #5a2040 !important;
}

/* ── Breathing widget ── */
.breath-circle {
    width: 120px; height: 120px;
    border-radius: 50%;
    background: linear-gradient(135deg, #e8427c, #b07be8);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; font-size: 1.1rem;
    margin: 1rem auto;
    box-shadow: 0 0 30px rgba(232,66,124,0.3);
    animation: breathe-idle 4s ease-in-out infinite;
}
@keyframes breathe-idle {
    0%,100% { transform: scale(1); box-shadow: 0 0 20px rgba(232,66,124,0.2); }
    50% { transform: scale(1.1); box-shadow: 0 0 40px rgba(232,66,124,0.4); }
}

/* ── Section headers ── */
.section-header {
    font-weight: 800;
    font-size: 1rem;
    color: #e8427c;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.sub-header {
    font-weight: 700;
    font-size: 0.9rem;
    color: #c47a99;
    margin-bottom: 0.5rem;
}

/* ── Reaction text ── */
.reaction-text {
    font-size: 0.88rem;
    color: #a0527a;
    font-style: italic;
    text-align: center;
    padding: 0.75rem 1rem;
    background: rgba(252,228,236,0.5);
    border: 1.5px solid #fce4ec;
    border-radius: 14px;
    margin-top: 0.5rem;
    font-weight: 400;
    line-height: 1.6;
}

/* ── Journal prompt ── */
.journal-prompt {
    font-size: 0.95rem;
    color: #5a2040;
    font-style: italic;
    line-height: 1.7;
    padding: 1rem;
    background: rgba(123,142,240,0.08);
    border-left: 3px solid #7b8ef0;
    border-radius: 0 12px 12px 0;
    margin: 0.5rem 0;
}

/* ── Coping strategy item ── */
.coping-item {
    font-size: 0.9rem;
    color: #5a2040;
    padding: 0.6rem 0.8rem;
    background: rgba(96,200,160,0.08);
    border-radius: 10px;
    margin: 0.4rem 0;
    border-left: 3px solid #60c8a0;
    line-height: 1.5;
}

/* ── Streak badge ── */
.streak-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: linear-gradient(135deg, #f0a860, #e8427c);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 1px;
}

/* ── Pet animations ── */
@keyframes float { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-10px); } }
@keyframes floatSlow { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-5px); } }
@keyframes bounce {
    0%,100% { transform: translateY(0px) scale(1); }
    25% { transform: translateY(-14px) scale(1.03); }
    50% { transform: translateY(-6px) scale(0.98); }
    75% { transform: translateY(-12px) scale(1.02); }
}
@keyframes shake {
    0%,100% { transform: rotate(0deg); }
    15% { transform: rotate(-5deg); }
    30% { transform: rotate(5deg); }
    45% { transform: rotate(-4deg); }
    60% { transform: rotate(4deg); }
}
@keyframes wobble {
    0%,100% { transform: translateX(0); }
    20% { transform: translateX(-5px); }
    40% { transform: translateX(5px); }
    60% { transform: translateX(-3px); }
    80% { transform: translateX(3px); }
}
@keyframes bow-bounce {
    0%,100% { transform: scale(1) rotate(-3deg); }
    50% { transform: scale(1.2) rotate(3deg); }
}
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 6px 30px rgba(232,66,124,0.2); }
    50% { box-shadow: 0 10px 50px rgba(232,66,124,0.45); }
}

.pet-joy     { animation: bounce 1.4s ease-in-out infinite; }
.pet-sad     { animation: floatSlow 6s ease-in-out infinite; }
.pet-angry   { animation: shake 0.5s ease-in-out infinite; }
.pet-fear    { animation: wobble 0.8s ease-in-out infinite; }
.pet-neutral { animation: float 3.5s ease-in-out infinite; }
.pet-img     { animation: float 3.5s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
defaults = {
    "mood_history":       [],
    "pet_name":           "Luna",
    "interaction_count":  0,
    "last_emotion":       None,
    "last_response":      None,
    "last_reaction":      None,
    "last_emotion_color": None,
    "last_confidence":    None,
    "last_journal_prompt": None,
    "last_coping":        None,
    "daily_streak":       0,
    "last_chat_date":     None,
    "show_breathing":     False,
    "show_journal":       False,
    "show_coping":        False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    path = Path("models/emotion_model.pkl")
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}

# ==================== STREAK LOGIC ====================
def update_streak():
    today = str(date.today())
    if st.session_state.last_chat_date != today:
        if st.session_state.last_chat_date is not None:
            st.session_state.daily_streak += 1
        else:
            st.session_state.daily_streak = 1
        st.session_state.last_chat_date = today

# ==================== KITTY SVG ====================
def generate_kitty_svg(emotion: str, size: int = 180, cat_png_path: str = "cat.png") -> str:
    cat_png = Path(cat_png_path)
    if cat_png.exists():
        img_b64 = get_base64_image(cat_png)
        if img_b64:
            anim_class = {
                "joy": "pet-joy", "anger": "pet-angry",
                "sadness": "pet-sad", "fear": "pet-fear",
            }.get(emotion, "pet-neutral")
            bow_emoji = {
                "joy": "🎀✨", "sadness": "💙", "anger": "💢",
                "fear": "💜", "surprise": "✨", "disgust": "🎀", "neutral": "🌸",
            }.get(emotion, "🌸")
            return f"""
            <div style="display:flex;flex-direction:column;align-items:center;gap:0.5rem;">
                <div class="{anim_class}" style="position:relative;display:inline-block;">
                    <img src="data:image/png;base64,{img_b64}"
                         width="{size}" height="{size}"
                         style="border-radius:50%;object-fit:cover;
                                border:3px solid #f8bbd0;
                                box-shadow:0 6px 30px rgba(232,66,124,0.2);
                                animation:pulse-glow 4s ease-in-out infinite;" />
                    <div style="position:absolute;top:-8px;right:-4px;font-size:1.2rem;
                                animation:bow-bounce 2s ease-in-out infinite;">
                        {bow_emoji}
                    </div>
                </div>
            </div>
            """

    # Fallback SVG
    body_colors = {
        "joy":      ("#ffe0ee", "#f8bbd0"), "sadness": ("#dce8ff", "#b0c8f0"),
        "anger":    ("#ffd0d0", "#ffb0b0"), "fear":    ("#e8d8f8", "#c8a8f0"),
        "surprise": ("#fff0d0", "#ffd890"), "disgust": ("#d0f0e0", "#a0ddc0"),
        "neutral":  ("#fce4ec", "#f8bbd0"),
    }
    bc1, bc2 = body_colors.get(emotion, body_colors["neutral"])
    bow_color = {
        "joy": "#e8427c", "sadness": "#7b8ef0", "anger": "#f06060",
        "fear": "#b07be8", "surprise": "#f0a060", "disgust": "#60c8a0",
        "neutral": "#e8427c",
    }.get(emotion, "#e8427c")

    mouth_d = {
        "joy":      "M 82 110 Q 100 122 118 110",
        "sadness":  "M 82 116 Q 100 106 118 116",
        "anger":    "M 85 112 Q 100 109 115 112",
        "fear":     "M 88 110 Q 100 118 112 110",
        "surprise": "M 94 108 Q 100 120 106 108",
        "disgust":  "M 84 114 Q 94 108 112 111",
        "neutral":  "M 88 112 Q 100 116 112 112",
    }.get(emotion, "M 88 112 Q 100 116 112 112")

    if emotion == "sadness":
        eye_html = """<ellipse cx="78" cy="88" rx="5" ry="4" fill="#3d1a2e"/>
        <ellipse cx="122" cy="88" rx="5" ry="4" fill="#3d1a2e"/>
        <line x1="74" y1="83" x2="82" y2="86" stroke="#3d1a2e" stroke-width="2" stroke-linecap="round"/>
        <line x1="118" y1="86" x2="126" y2="83" stroke="#3d1a2e" stroke-width="2" stroke-linecap="round"/>"""
    elif emotion == "anger":
        eye_html = """<ellipse cx="78" cy="88" rx="5" ry="5" fill="#3d1a2e"/>
        <ellipse cx="122" cy="88" rx="5" ry="5" fill="#3d1a2e"/>
        <line x1="72" y1="82" x2="84" y2="86" stroke="#3d1a2e" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="116" y1="86" x2="128" y2="82" stroke="#3d1a2e" stroke-width="2.5" stroke-linecap="round"/>"""
    elif emotion in ("surprise", "fear"):
        eye_html = """<ellipse cx="78" cy="88" rx="8" ry="8" fill="#3d1a2e"/>
        <ellipse cx="122" cy="88" rx="8" ry="8" fill="#3d1a2e"/>
        <circle cx="82" cy="84" r="3" fill="white"/>
        <circle cx="126" cy="84" r="3" fill="white"/>"""
    else:
        eye_html = """<ellipse cx="78" cy="88" rx="5" ry="5.5" fill="#3d1a2e"/>
        <ellipse cx="122" cy="88" rx="5" ry="5.5" fill="#3d1a2e"/>
        <circle cx="80" cy="85" r="2" fill="white"/>
        <circle cx="124" cy="85" r="2" fill="white"/>"""

    blush_html = """<ellipse cx="72" cy="108" rx="11" ry="7" fill="#f48fb1" opacity="0.4"/>
    <ellipse cx="128" cy="108" rx="11" ry="7" fill="#f48fb1" opacity="0.4"/>""" if emotion in ("joy", "surprise", "neutral") else ""
    tear_html = """<ellipse cx="78" cy="96" rx="2.5" ry="5" fill="#90caf9" opacity="0.75"/>
    <ellipse cx="122" cy="96" rx="2.5" ry="5" fill="#90caf9" opacity="0.75"/>""" if emotion == "sadness" else ""
    sparkle_html = """<text x="148" y="52" font-size="14" opacity="0.9">✨</text>
    <text x="18" y="58" font-size="12" opacity="0.7">🌸</text>""" if emotion == "joy" else ""

    anim_class = {
        "joy": "pet-joy", "anger": "pet-angry",
        "sadness": "pet-sad", "fear": "pet-fear",
    }.get(emotion, "pet-neutral")

    return f"""
    <div style="display:flex;justify-content:center;" class="{anim_class}">
    <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
      <defs>
        <radialGradient id="faceG" cx="50%" cy="38%" r="58%">
          <stop offset="0%" stop-color="white"/>
          <stop offset="100%" stop-color="{bc1}"/>
        </radialGradient>
      </defs>
      <ellipse cx="62" cy="48" rx="22" ry="26" fill="{bc2}"/>
      <ellipse cx="138" cy="48" rx="22" ry="26" fill="{bc2}"/>
      <ellipse cx="62" cy="48" rx="14" ry="18" fill="white" opacity="0.6"/>
      <ellipse cx="138" cy="48" rx="14" ry="18" fill="white" opacity="0.6"/>
      <circle cx="100" cy="108" r="72" fill="url(#faceG)" stroke="{bc2}" stroke-width="1.5"/>
      <polygon points="80,42 62,30 68,52" fill="{bow_color}" opacity="0.9"/>
      <polygon points="80,42 98,30 92,52" fill="{bow_color}" opacity="0.9"/>
      <circle cx="80" cy="42" r="7" fill="{bow_color}"/>
      <polygon points="80,42 62,30 68,52" fill="white" opacity="0.2"/>
      <polygon points="80,42 98,30 92,52" fill="white" opacity="0.2"/>
      {eye_html}
      <ellipse cx="100" cy="100" rx="5" ry="4" fill="#f48fb1"/>
      <line x1="22" y1="98" x2="70" y2="102" stroke="#ccc" stroke-width="1.2" opacity="0.7"/>
      <line x1="22" y1="108" x2="70" y2="108" stroke="#ccc" stroke-width="1.2" opacity="0.7"/>
      <line x1="130" y1="102" x2="178" y2="98" stroke="#ccc" stroke-width="1.2" opacity="0.7"/>
      <line x1="130" y1="108" x2="178" y2="108" stroke="#ccc" stroke-width="1.2" opacity="0.7"/>
      <path d="{mouth_d}" stroke="#c0607a" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      {blush_html}{tear_html}{sparkle_html}
    </svg>
    </div>"""


# ==================== BREATHING WIDGET ====================
def show_breathing_widget(emotion: str):
    exercise = get_breathing_exercise(emotion)
    st.markdown(f'<div class="card-breathe">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">🌬️ {exercise["name"]}</div>', unsafe_allow_html=True)
    st.caption(exercise["description"])

    steps = exercise["steps"]
    step_labels = " → ".join([f"**{s['phase']}** ({s['duration']}s)" for s in steps])
    st.markdown(step_labels)

    st.markdown('<div class="breath-circle">Breathe</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cycles = st.number_input("Cycles", min_value=1, max_value=10, value=exercise["cycles"], key="breath_cycles")
    with col2:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if st.button("▶ Start Breathing", key="start_breath"):
            with st.empty():
                for cycle in range(cycles):
                    for step in steps:
                        for remaining in range(step["duration"], 0, -1):
                            st.markdown(f"""
                            <div style="text-align:center;padding:1rem;">
                                <div style="font-size:1.5rem;font-weight:800;color:#e8427c;">{step['phase']}</div>
                                <div style="font-size:3rem;font-weight:800;color:#b07be8;">{remaining}</div>
                                <div style="color:#a0527a;font-style:italic;">{step['instruction']}</div>
                                <div style="color:#c47a99;font-size:0.8rem;margin-top:0.5rem;">Cycle {cycle+1}/{cycles}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            time.sleep(1)
                st.markdown('<div style="text-align:center;font-size:1.2rem;color:#e8427c;padding:1rem;">✨ Complete! Well done 🌸</div>', unsafe_allow_html=True)
                time.sleep(2)

    st.markdown('</div>', unsafe_allow_html=True)


# ==================== MAIN ====================
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.session_state.pet_name = st.text_input(
            "Pet's name 🌸", value=st.session_state.pet_name, max_chars=15
        )
        st.divider()

       # ==================== MODEL INFO & FEATURES ====================
        model = load_model()
        if model:
            ver = model.get("version", "?")
            acc = model.get("accuracy", 0)
            st.markdown(f"**Model:** v{ver} | **Accuracy:** {acc:.1%}")
            if model.get("ensemble"):
                st.markdown("🔧 *Ensemble mode (SVC + LR)*")
        else:
            st.warning("No trained model found. Run `train_model.py` first.")

        st.divider()

        # --- Feature Description ---
        st.markdown(f"""
        ### 🐾 Meet {st.session_state.pet_name}
        I use an **Ensemble ML model** to sense how you're feeling through your words.
        
        **I can recognize:**
        - **Joy** 
        - **Sadness** 
        - **Anger** 
        - **Fear** 
        
        **I can help with:**
        -  Guided breathing
        -  Journaling prompts
        -  Coping strategies
        """)

        st.divider()

        # --- History Management ---
        if st.button("🗑️ Clear history"):
            for k in ["mood_history","interaction_count","last_emotion","last_response",
                      "last_reaction","last_confidence","last_journal_prompt","last_coping",
                      "show_breathing","show_journal","show_coping"]:
                st.session_state[k] = [] if k == "mood_history" else (
                    0 if k == "interaction_count" else (False if "show" in k else None))
            st.success("Cleared! ✨")
            st.rerun()

        if st.session_state.mood_history:
            csv = "Time,Emotion,Confidence,Text\n"
            for e in st.session_state.mood_history:
                csv += f"{e['time']},{e['emotion']},{e.get('confidence','N/A')},\"{e['text']}\"\n"
            st.download_button("📥 Export Mood CSV", csv, "mood_history.csv", "text/csv")

        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #c47a99; font-size: 0.8rem;">
            <b>Comfort Pet</b> 🌸<br>
            Designed by Ipshita Bhardwaj
        </div>
        """, unsafe_allow_html=True)

    # Load model for the main page logic
    model = load_model()
    # ── Header ───────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<span class="bow-accent">🎀</span>', unsafe_allow_html=True)
        st.markdown(f'<h1 class="site-title">{st.session_state.pet_name}</h1>', unsafe_allow_html=True)
        st.markdown('<p class="site-sub">your emotional companion ✨</p>', unsafe_allow_html=True)

    with col_h2:
        stats_html = ""
        if st.session_state.interaction_count > 0:
            stats_html += f"""
            <div class="stat-card" style="margin-top:1.5rem;margin-bottom:0.5rem;">
                <div class="stat-num">{st.session_state.interaction_count}</div>
                <div class="stat-label">chats today</div>
            </div>"""
        if st.session_state.daily_streak > 1:
            stats_html += f'<div style="text-align:center;margin-top:0.4rem;"><span class="streak-badge">🔥 {st.session_state.daily_streak} day streak</span></div>'
        if stats_html:
            st.markdown(stats_html, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Main layout ───────────────────────────────────────────────────────
    pet_col, chat_col = st.columns([1, 1.65], gap="large")

    with pet_col:
        st.markdown('<div class="card card-pet">', unsafe_allow_html=True)
        emotion_now = st.session_state.last_emotion or "neutral"

        st.markdown('<div class="pet-area">', unsafe_allow_html=True)
        st.markdown(f'<div class="pet-name">{st.session_state.pet_name} 🌸</div>', unsafe_allow_html=True)
        st.markdown(generate_kitty_svg(emotion_now, size=175), unsafe_allow_html=True)

        mood_labels = {
            "joy":      "So happy for you! ✨",
            "sadness":  "Here for you 💙",
            "anger":    "Breathing with you 🌬️",
            "fear":     "You're safe with me 🛡️",
            "surprise": "Wide-eyed together! 👀",
            "disgust":  "Your gut is right 🎯",
            "neutral":  "Ready to listen 🌸",
        }
        st.markdown(f'<div class="pet-mood-text">{mood_labels.get(emotion_now, "Ready to listen 🌸")}</div>', unsafe_allow_html=True)

        if st.session_state.last_reaction:
            st.markdown(f'<div class="reaction-text">{st.session_state.last_reaction}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Quick actions for emotional support ───────────────────────────
        if st.session_state.last_emotion and st.session_state.last_emotion in ("fear", "anger", "sadness"):
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🌬️ Breathe", key="btn_breathe"):
                    st.session_state.show_breathing = not st.session_state.show_breathing
                    st.session_state.show_journal   = False
                    st.session_state.show_coping    = False
            with b2:
                if st.button("🛡️ Cope", key="btn_cope"):
                    st.session_state.show_coping    = not st.session_state.show_coping
                    st.session_state.show_breathing = False
                    st.session_state.show_journal   = False

        if st.session_state.last_emotion:
            if st.button("✍️ Journal Prompt", key="btn_journal"):
                st.session_state.show_journal    = not st.session_state.show_journal
                st.session_state.show_breathing  = False
                st.session_state.show_coping     = False
                if st.session_state.show_journal:
                    st.session_state.last_journal_prompt = get_journal_prompt(st.session_state.last_emotion)
                    st.session_state.last_coping = get_coping_strategies(st.session_state.last_emotion)

    with chat_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">💭 How are you feeling?</div>', unsafe_allow_html=True)

        user_input = st.text_area(
            "input",
            placeholder="Share what's on your mind… I'm listening 🌸",
            height=120,
            label_visibility="collapsed",
            key="user_input_area"
        )

        submit = st.button(
            f"Share with {st.session_state.pet_name} 🎀",
            use_container_width=True
        )

        if submit and user_input.strip():
            update_streak()
            st.session_state.interaction_count += 1
            emotion, confidence = detect_emotion_with_confidence(user_input, model)
            color    = get_emotion_color(emotion)
            response = get_comfort_response(emotion, user_input)
            reaction = get_pet_reaction(emotion)
            journal_prompt = get_journal_prompt(emotion)
            coping = get_coping_strategies(emotion)

            st.session_state.last_emotion        = emotion
            st.session_state.last_response       = response
            st.session_state.last_reaction       = reaction
            st.session_state.last_emotion_color  = color
            st.session_state.last_confidence     = confidence
            st.session_state.last_journal_prompt = journal_prompt
            st.session_state.last_coping         = coping
            st.session_state.show_breathing      = False
            st.session_state.show_journal        = False
            st.session_state.show_coping         = False

            st.session_state.mood_history.append({
                "time":       datetime.now().strftime("%H:%M"),
                "emotion":    emotion,
                "confidence": confidence,
                "text":       (user_input[:55] + "…") if len(user_input) > 55 else user_input,
                "full_text":  user_input,
                "timestamp":  datetime.now(),
            })
            if len(st.session_state.mood_history) > 30:
                st.session_state.mood_history = st.session_state.mood_history[-30:]

        elif submit:
            st.warning("Please write something first! 🌸")

        # ── Response area ──────────────────────────────────────────────
        if st.session_state.last_response:
            color      = st.session_state.last_emotion_color or "#e8427c"
            conf       = st.session_state.last_confidence or 0
            emoji_char = get_emotion_emoji(st.session_state.last_emotion or "neutral")
            g = EMOTION_GRADIENTS.get(st.session_state.last_emotion or "neutral", ("e8427c", "f48fb1"))

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="emotion-pill"
                 style="background:{color}18;color:{color};border-color:{color}44;">
                {emoji_char}&nbsp;{(st.session_state.last_emotion or 'neutral').upper()}
                &nbsp;<span style="opacity:0.65;font-size:0.75rem;font-weight:600">{conf}% confident</span>
            </div>
            <div class="conf-wrap">
                <div class="conf-fill" style="width:{conf}%;background:linear-gradient(90deg,#{g[0]},#{g[1]});"></div>
            </div>
            """, unsafe_allow_html=True)

            if submit and user_input.strip():
                placeholder = st.empty()
                displayed = ""
                for char in st.session_state.last_response:
                    displayed += char
                    placeholder.markdown(
                        f'<div class="response-text">{displayed}</div>',
                        unsafe_allow_html=True
                    )
                    time.sleep(0.007)
            else:
                st.markdown(
                    f'<div class="response-text">{st.session_state.last_response}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Breathing / Journal / Coping panels ──────────────────────────────
    if st.session_state.show_breathing and st.session_state.last_emotion:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        show_breathing_widget(st.session_state.last_emotion)

    if st.session_state.show_coping and st.session_state.last_coping:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card-coping">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🛡️ Coping Strategies</div>', unsafe_allow_html=True)
        for strategy in st.session_state.last_coping:
            st.markdown(f'<div class="coping-item">{strategy}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.show_journal and st.session_state.last_journal_prompt:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card-journal">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">✍️ Journal Prompt</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="journal-prompt">"{st.session_state.last_journal_prompt}"</div>', unsafe_allow_html=True)
        st.text_area("Your thoughts...", placeholder="Write freely here — no judgment, just you. 🌸",
                     height=120, key="journal_text")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Mood history ──────────────────────────────────────────────────────
    if st.session_state.mood_history:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["🌸  History", "📈  Insights", "📊  Chart", "🔍  Details"])

        with tab1:
            for entry in reversed(st.session_state.mood_history):
                color = get_emotion_color(entry["emotion"])
                emoji = get_emotion_emoji(entry["emotion"])
                conf  = entry.get("confidence", "")
                st.markdown(f"""
                <div class="mood-entry" style="border-left-color:{color};">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span class="mood-entry-emotion" style="color:{color};">{emoji} {entry['emotion'].upper()}</span>
                        <span class="mood-entry-time">{entry['time']} · {conf}%</span>
                    </div>
                    <div class="mood-entry-text">"{entry['text']}"</div>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            trend = analyze_mood_trend(st.session_state.mood_history)
            st.info(trend)

            ec: dict = {}
            for e in st.session_state.mood_history:
                ec[e["emotion"]] = ec.get(e["emotion"], 0) + 1

            if ec:
                cols = st.columns(min(len(ec), 4))
                for idx, (emo, cnt) in enumerate(sorted(ec.items())):
                    with cols[idx % 4]:
                        pct = cnt / len(st.session_state.mood_history) * 100
                        color = get_emotion_color(emo)
                        st.markdown(f"""
                        <div class="stat-card">
                            <div style="font-size:1.4rem;">{get_emotion_emoji(emo)}</div>
                            <div class="stat-num" style="font-size:1.4rem;color:{color};">{cnt}</div>
                            <div class="stat-label">{emo}</div>
                            <div style="font-size:0.72rem;color:#c47a99;margin-top:0.15rem;font-weight:600">{pct:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

        with tab3:
            # Simple ASCII-style emotion timeline using HTML bars
            if len(st.session_state.mood_history) >= 2:
                emotion_order = ["joy", "surprise", "neutral", "disgust", "fear", "sadness", "anger"]
                st.markdown('<div class="sub-header">📉 Mood Timeline (recent → oldest)</div>', unsafe_allow_html=True)
                recent = list(reversed(st.session_state.mood_history[-15:]))
                bar_html = '<div style="display:flex;gap:4px;align-items:flex-end;height:120px;padding:0.5rem 0;">'
                for entry in recent:
                    emo = entry["emotion"]
                    color = get_emotion_color(emo)
                    emoji = get_emotion_emoji(emo)
                    conf = entry.get("confidence", 50)
                    height = max(20, int(conf * 1.1))
                    bar_html += f'''<div style="display:flex;flex-direction:column;align-items:center;gap:2px;flex:1;" title="{emo} {conf}%">
                        <div style="font-size:0.7rem;color:{color};font-weight:700;">{conf}%</div>
                        <div style="background:{color};height:{height}px;width:100%;border-radius:4px 4px 0 0;opacity:0.85;"></div>
                        <div style="font-size:0.85rem;" title="{emo}">{emoji}</div>
                    </div>'''
                bar_html += '</div>'
                st.markdown(bar_html, unsafe_allow_html=True)
                st.caption("Bar height = detection confidence. Colors map to emotions.")

                # Emotion frequency donut-style via simple HTML
                st.markdown('<div class="sub-header" style="margin-top:1rem;">🎯 Emotion Frequency</div>', unsafe_allow_html=True)
                ec2: dict = {}
                for e in st.session_state.mood_history:
                    ec2[e["emotion"]] = ec2.get(e["emotion"], 0) + 1
                total = sum(ec2.values())
                freq_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:0.5rem;">'
                for emo, cnt in sorted(ec2.items(), key=lambda x: -x[1]):
                    pct = cnt / total * 100
                    color = get_emotion_color(emo)
                    emoji = get_emotion_emoji(emo)
                    freq_html += f'<div style="background:{color}20;border:2px solid {color}50;border-radius:20px;padding:4px 12px;font-size:0.82rem;font-weight:700;color:{color};">{emoji} {emo} {pct:.0f}%</div>'
                freq_html += '</div>'
                st.markdown(freq_html, unsafe_allow_html=True)
            else:
                st.info("Chat more to see your mood chart! 🌸")

        with tab4:
            for entry in reversed(st.session_state.mood_history):
                with st.expander(f"{get_emotion_emoji(entry['emotion'])}  {entry['emotion'].upper()} — {entry['time']}"):
                    st.write(f"**Message:** {entry['full_text']}")
                    st.write(f"**Detected Emotion:** {entry['emotion'].title()}")
                    st.write(f"**Confidence:** {entry.get('confidence', 'N/A')}%")
                    st.write(f"**Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Footer ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;color:#d4a0b8;font-size:0.78rem;padding:3rem 0 1rem;font-weight:600;">
        Made with love 🌸🎀 · Ensemble ML (SVC + LR) + Rule-Based
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()