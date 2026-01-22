# ==========================================================
# Sora Prompt Studio Pro – Director Edition (FINAL STABLE)
# PROMPT 1 + PROMPT 2 (CAMEO FROM START • 3 LINES GUARANTEED)
# ==========================================================

import streamlit as st
import pandas as pd
import random
import re
import json
import unicodedata
from pathlib import Path
from typing import List, Tuple
from PIL import Image

# =========================
# BASIC CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro – Director Edition (FINAL)")
st.caption("Prompt 1 & 2 • 10s • Anti-duplicate • Copy Safe • Cameo from start • Disclaimer guaranteed")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

REQUIRED_FILES = [
    "dialogue_library.csv",
    "scene_library.csv",
    "disclaimer_prompt2.csv",
]

# =========================
# TEXT SAFE
# =========================
ZERO_WIDTH = r"[\u200b\u200c\u200d\uFEFF]"

def normalize(s):
    if not s:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = re.sub(ZERO_WIDTH, "", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join([x.rstrip() for x in s.split("\n")]).strip()

def ensure_dot(s):
    s = normalize(s)
    if not s:
        return ""
    if s.endswith((".", "!", "?")):
        return s
    return s + "."

# =========================
# COPY SAFE BUTTON
# =========================
def copy_btn(text, key):
    text = normalize(text)
    payload = json.dumps(text)
    html = f"""
    <button id="{key}" style="padding:8px 14px;border-radius:8px;border:1px solid #ccc;font-weight:700;">
    COPY
    </button>
    <script>
    const t = {payload};
    document.getElementById("{key}").onclick = async () => {{
        await navigator.clipboard.writeText(t);
    }};
    </script>
    """
    st.components.v1.html(html, height=42)

# =========================
# FILE LOAD
# =========================
missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing))
    st.stop()

@st.cache_data
def read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except:
        return pd.read_csv(path, encoding="utf-8", errors="replace")

dialogues = read_csv("dialogue_library.csv").to_dict(orient="records")
scenes = read_csv("scene_library.csv").to_dict(orient="records")
disclaimers = read_csv("disclaimer_prompt2.csv").iloc[:, -1].dropna().astype(str).tolist()

# =========================
# SESSION – ANTI DUP
# =========================
if "used_dialogues" not in st.session_state:
    st.session_state.used_dialogues = set()
if "used_scenes" not in st.session_state:
    st.session_state.used_scenes = set()
if "outputs" not in st.session_state:
    st.session_state.outputs = []

def pick_unique(pool, used, key):
    fresh = [x for x in pool if str(x.get(key)) not in used]
    if not fresh:
        used.clear()
        fresh = pool[:]
    item = random.choice(fresh)
    used.add(str(item.get(key)))
    return item

# =========================
# STYLE PACK
# =========================
STYLE_PACK = [
    {"id":"s1","name":"Bright boutique studio","lens":"40-50mm","camera":"smooth orbit, ultra sharp"},
    {"id":"s2","name":"Daylight cafe","lens":"35mm","camera":"handheld stable"},
    {"id":"s3","name":"Modern street morning","lens":"28mm","camera":"tracking low"},
    {"id":"s4","name":"Penthouse window light","lens":"50mm","camera":"slow pan"},
    {"id":"s5","name":"Showroom macro","lens":"85mm","camera":"macro ultra sharp"},
]

VOICE_STYLES = [
    "Calm, slow luxury pacing; warm confident male voice",
    "Soft storytelling; relaxed tempo; intimate phone-video feel",
    "Low deep voice; steady tempo; premium showroom vibe",
    "Bright but controlled; friendly and natural tone",
    "Energetic clean voice; slightly faster tempo but clear",
]

# =========================
# UI
# =========================
left, right = st.columns([1.1, 0.9])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg","png","jpeg"])
    mode_ui = st.radio("Prompt mode", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone = st.selectbox("Tone", ["Tự nhiên","Tự tin","Lãng mạn","Mạnh mẽ","Truyền cảm"], index=0)
    shot_count = st.slider("Shots (2–4)", 2, 4, 4)
    batch_count = st.slider("Prompts per click", 1, 10, 5)

with right:
    st.write("PROMPT 2:")
    st.write("• Cameo xuất hiện từ 0.0s cùng giày")
    st.write("• 3 dòng thoại BẮT BUỘC")
    st.write("• Câu 3 luôn là miễn trừ")
    st.write("• Ép đọc đủ 1.2–6.9s")

# =========================
# PROMPT BUILDER
# =========================
SEP = "══════════════════════════════════"

def build_prompt(mode, shoe_name, shoe_type, style, scene_list, voice_lines, voice_style):

    if mode == "p1":
        title = "VIDEO SETUP — SLOW LUXURY EDITION (NO CAMEO)"
        cast = f"NO people on screen\nNO cameo visible\nVOICE ID: {CAMEO_VOICE_ID}"
    else:
        title = "VIDEO SETUP — SLOW LUXURY EDITION (WITH CAMEO FROM START)"
        cast = (
            "CAST RULE — PROMPT 2\n"
            "Cameo appears from 0.0s and stays visible the whole video.\n"
            "Cameo holds the shoe naturally, face and shoe both clear.\n"
            f"CAMEO & VOICE ID: {CAMEO_VOICE_ID}\n"
            "No price, no discount, no guarantees"
        )

    shots = []
    t = [0.0, 2.5, 5.0, 7.5, 10.0]
    for i, sc in enumerate(scene_list):
        shots.append(f"{t[i]:.1f}–{t[i+1]:.1f}s: {sc.get('location','scene')}. Camera {sc.get('motion','pan')}.")

    shot_block = "\n".join(shots)

    prompt = f"""
{CAMEO_VOICE_ID}

{SEP}
{title}
{SEP}
Video dọc 9:16 — 10s
Ultra Sharp PRO 4K
Use the EXACT uploaded shoe image
TikTok-safe absolute

NO text
NO logo
NO watermark
NO blur

{SEP}
VISUAL STYLE
{SEP}
Style: {style['name']}
Lens: {style['lens']}
Camera: {style['camera']}

{SEP}
CAST RULE
{SEP}
{cast}

{SEP}
PRODUCT
{SEP}
shoe_name: {shoe_name}
shoe_type: {shoe_type}

{SEP}
SHOT LIST
{SEP}
{shot_block}

{SEP}
AUDIO MASTERING — PROMPT {mode.upper()}
{SEP}
0.0–1.2s: NO voice
1.2–3.2s: Line 1
3.2–5.3s: Line 2
5.3–6.9s: Line 3 (DISCLAIMER) — MUST be spoken
6.9–10.0s: Voice OFF, music only

{SEP}
VOICEOVER (1.2–6.9s)
{SEP}
{voice_lines}

{SEP}
HARD RULES
{SEP}
NO mirrored logo
NO reversed letters
NO shoe distortion
""".strip()

    return normalize(prompt)

# =========================
# GENERATE
# =========================
if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_"," ")
    img = Image.open(uploaded)
    st.image(img, caption=uploaded.name, use_container_width=True)

    if st.button("GENERATE PROMPTS", use_container_width=True):

        outputs = []
        used_styles = set()

        for _ in range(batch_count):

            style = random.choice(STYLE_PACK)
            scene_list = random.sample(scenes, min(shot_count, len(scenes)))

            # -------- VOICE (ANTI DUP ABSOLUTE)
            d = pick_unique(dialogues, st.session_state.used_dialogues, "id")

            lines = re.split(r"[.!?]+", d.get("dialogue",""))
            lines = [ensure_dot(x) for x in lines if x.strip()]

            if mode_ui.startswith("PROMPT 1"):
                l1 = lines[0] if len(lines)>0 else "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
                l2 = lines[1] if len(lines)>1 else "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định."
                l3 = lines[2] if len(lines)>2 else "Nhìn tổng thể tối giản nhưng vẫn có điểm tinh tế."
                voice_lines = f"{l1}\n{l2}\n{l3}"
                mode = "p1"
            else:
                l1 = lines[0] if len(lines)>0 else "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
                l2 = lines[1] if len(lines)>1 else "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định."
                disc = ensure_dot(random.choice(disclaimers))
                voice_lines = f"{l1}\n{l2}\n{disc}"
                mode = "p2"

            prompt = build_prompt(
                mode=mode,
                shoe_name=shoe_name,
                shoe_type="leather",
                style=style,
                scene_list=scene_list,
                voice_lines=voice_lines,
                voice_style=random.choice(VOICE_STYLES),
            )

            outputs.append(prompt)

        st.session_state.outputs = outputs

# =========================
# OUTPUT
# =========================
if st.session_state.outputs:
    tabs = st.tabs([f"Prompt {i+1}" for i in range(len(st.session_state.outputs))])
    for i, tab in enumerate(tabs):
        with tab:
            st.text_area("Prompt", st.session_state.outputs[i], height=600)
            copy_btn(st.session_state.outputs[i], f"copy{i}")

# =========================
# RESET
# =========================
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogues.clear()
    st.session_state.used_scenes.clear()
    st.session_state.outputs = []
    st.success("Reset done.")
