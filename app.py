import streamlit as st
import pandas as pd
import random
import re
import unicodedata
from pathlib import Path

# ==========================================================
# SORA PROMPT STUDIO PRO – STABLE VERSION (SORA SAFE)
# - No Gemini (no quota error)
# - Old working prompt format
# - 4 shots / 10s
# - 5 prompts per click
# - Unicode safe copy
# ==========================================================

st.set_page_config(page_title="Sora Prompt Studio Pro - Stable", layout="wide")
st.title("Sora Prompt Studio Pro – Stable Edition")
st.caption("Prompt chuẩn Sora • 10s • Multi-shot • Copy không lỗi dấu • Render ổn định")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

# =========================
# TEXT NORMALIZE
# =========================
ZERO_WIDTH_PATTERN = r"[\u200b\u200c\u200d\uFEFF]"

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    try:
        s = unicodedata.normalize("NFC", s)
    except Exception:
        pass
    s = re.sub(ZERO_WIDTH_PATTERN, "", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join([line.rstrip() for line in s.split("\n")])
    return s.strip()

def ensure_end_punct(s: str) -> str:
    s = normalize_text(s)
    if not s:
        return ""
    if s.endswith((".", "!", "?")):
        return s
    return s + "."

# =========================
# FILE CHECK
# =========================
missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing))
    st.stop()

# =========================
# LOAD CSV
# =========================
@st.cache_data
def load_dialogues():
    df = pd.read_csv("dialogue_library.csv", encoding="utf-8-sig")
    return df.to_dict(orient="records")

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv", encoding="utf-8-sig")
    return df.to_dict(orient="records")

@st.cache_data
def load_disclaimers():
    df = pd.read_csv("disclaimer_prompt2.csv", encoding="utf-8-sig")
    col = df.columns[-1]
    return [normalize_text(x) for x in df[col].dropna().astype(str).tolist()]

dialogues = load_dialogues()
scenes = load_scenes()
disclaimers = load_disclaimers()

# =========================
# SESSION ANTI DUP
# =========================
if "used_dialogue" not in st.session_state:
    st.session_state.used_dialogue = set()
if "used_scene" not in st.session_state:
    st.session_state.used_scene = set()
if "generated" not in st.session_state:
    st.session_state.generated = []

# =========================
# UTILS
# =========================
def pick_unique(pool, used_set, key):
    def get_id(x):
        return str(x.get(key, "")) + str(hash(str(x)))

    available = [x for x in pool if get_id(x) not in used_set]
    if not available:
        used_set.clear()
        available = pool[:]

    item = random.choice(available)
    used_set.add(get_id(item))
    return item

def split_timeline():
    cuts = [0.0, 2.5, 5.0, 7.5, 10.0]
    return [(cuts[i], cuts[i+1]) for i in range(4)]

# =========================
# PROMPT BUILDER (FORMAT CŨ – SORA RENDER ỔN)
# =========================
SEP = "══════════════════════════════════"

def build_prompt(shoe_name, shoe_type, scene_list, voice_lines):
    shot_lines = []
    timeline = split_timeline()

    for sc, (a, b) in zip(scene_list, timeline):
        loc = normalize_text(sc.get("location", "studio sáng"))
        light = normalize_text(sc.get("lighting", "ánh sáng trắng sạch"))
        mot = normalize_text(sc.get("motion", "slow pan"))
        wea = normalize_text(sc.get("weather", "không mưa"))
        mood = normalize_text(sc.get("mood", "premium"))
        line = f"{a:.1f}–{b:.1f}s: {loc}. {light}. Camera {mot}. Weather {wea}. Mood {mood}."
        shot_lines.append(ensure_end_punct(line))

    shot_block = "\n".join(shot_lines)

    prompt = f"""
{CAMEO_VOICE_ID}

{SEP}
VIDEO SETUP — SLOW LUXURY EDITION (NO CAMEO) (FINAL • TEXT ORIENTATION & SHARPNESS LOCK)
{SEP}
Video dọc 9:16 — 10s
Ultra Sharp PRO 4K output (internal 12K)
Realistic cinematic video (NOT static image)
Use the EXACT uploaded shoe image
TikTok-safe absolute

NO text
NO logo
NO watermark
NO blur
NO haze
NO glow

{SEP}
ULTRA BRIGHTNESS + SHARPNESS LOCK — ABSOLUTE
{SEP}
Bright exposure, HDR+, clean blacks, no underexposure
Shoe is the sharpest object on screen in all frames
Zero motion blur on shoe
No foggy lighting, no darkness

{SEP}
SHOE REFERENCE — ABSOLUTE LOCK
{SEP}
Use ONLY the uploaded shoe image as reference
LOCK shoe identity: toe, vamp, stitching, sole, proportions
NO redesign
NO deformation
NO color shift
If laces exist → keep laces in ALL frames
If no laces → never add laces

{SEP}
TEXT & LOGO ORIENTATION LOCK — ABSOLUTE
{SEP}
If shoe has text or symbol:
NOT mirrored
NOT reversed
NOT flipped

{SEP}
PRODUCT
{SEP}
shoe_name: {shoe_name}
shoe_type: {shoe_type}

{SEP}
SHOT LIST — TOTAL 10s (MULTI-SHOT)
{SEP}
{shot_block}

{SEP}
AUDIO MASTERING
{SEP}
0.0–1.2s: NO voice
1.2–6.9s: VOICE ON
6.9–10.0s: NO voice, music only

{SEP}
VOICEOVER (1.2–6.9s)
{SEP}
{normalize_text(voice_lines)}

{SEP}
HARD RULES — ABSOLUTE
{SEP}
NO on-screen text
NO logos overlay
NO watermark
NO mirrored logo
NO reversed letters
NO incorrect shoe
""".strip()

    return normalize_text(prompt)

# =========================
# UI
# =========================
uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
tone = st.selectbox("Tone", ["Tự nhiên", "Tự tin", "Truyền cảm", "Mạnh mẽ"], index=0)
count = st.slider("Prompts per click", 1, 10, 5)
shoe_type = st.selectbox("Shoe type", SHOE_TYPES, index=SHOE_TYPES.index("leather"))

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    st.image(uploaded, caption=uploaded.name, use_container_width=True)

    if st.button("Generate", use_container_width=True):
        arr = []

        for _ in range(count):
            scene_list = [pick_unique(scenes, st.session_state.used_scene, "id") for _ in range(4)]
            d = pick_unique(dialogues, st.session_state.used_dialogue, "id")

            parts = re.split(r"[.!?]+", str(d.get("dialogue", "")))
            a = parts[0] if len(parts) > 0 else "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
            b = parts[1] if len(parts) > 1 else "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định."
            c = parts[2] if len(parts) > 2 else "Nhìn tổng thể tối giản nhưng vẫn có điểm tinh tế."

            voice = normalize_text(
                ensure_end_punct(a) + "\n" +
                ensure_end_punct(b) + "\n" +
                ensure_end_punct(c)
            )

            prompt = build_prompt(shoe_name, shoe_type, scene_list, voice)
            arr.append(prompt)

        st.session_state.generated = arr

    prompts = st.session_state.generated
    if prompts:
        st.markdown("### Output prompts")
        for i, p in enumerate(prompts):
            st.text_area(f"Prompt {i+1}", p, height=520)

else:
    st.warning("Upload shoe image to begin.")

st.divider()
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue.clear()
    st.session_state.used_scene.clear()
    st.session_state.generated = []
    st.success("Reset done.")
