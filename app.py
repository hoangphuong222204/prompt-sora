import streamlit as st
import pandas as pd
import random
import re
import json
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 - Total 10s - Multi-shot - Anti-duplicate - TikTok Shop SAFE - SORA ENGINE SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"

SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]
REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

# =========================
# TEXT NORMALIZE (COPY SAFE)
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
    return s.strip()

def safe_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = normalize_text(str(v))
    if s.lower() == "nan":
        return ""
    return s

def compact_spaces(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

# =========================
# COPY BUTTON (UNICODE SAFE)
# =========================
def copy_button_unicode_safe(text: str, key: str):
    text = normalize_text(text)
    payload = json.dumps(text)
    html = f"""
    <button id="{key}" style="
        padding:8px 14px;border-radius:10px;border:1px solid #ccc;
        cursor:pointer;background:#fff;font-weight:700;">COPY</button>
    <span id="{key}_s" style="margin-left:8px;font-size:12px;"></span>
    <script>
    (function() {{
        const btn = document.getElementById("{key}");
        const s = document.getElementById("{key}_s");
        const text = {payload};
        btn.onclick = async () => {{
            try {{
                await navigator.clipboard.writeText(text);
                s.innerText = "Copied";
                setTimeout(()=>s.innerText="",1500);
            }} catch(e) {{
                s.innerText = "Clipboard blocked";
                setTimeout(()=>s.innerText="",2500);
            }}
        }};
    }})();
    </script>
    """
    st.components.v1.html(html, height=42)

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
def read_csv_flexible(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8", errors="replace")

@st.cache_data
def load_dialogues():
    df = read_csv_flexible("dialogue_library.csv")
    df.columns = [c.strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_scenes():
    df = read_csv_flexible("scene_library.csv")
    df.columns = [c.strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_disclaimer_prompt2():
    df = read_csv_flexible("disclaimer_prompt2.csv")
    col = df.columns[-1]
    arr = df[col].dropna().astype(str).tolist()
    return [normalize_text(x) for x in arr if normalize_text(x)]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2()

# =========================
# SESSION – ANTI DUP
# =========================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []

# =========================
# UTILS
# =========================
def pick_unique(pool, used_ids: set, key: str):
    def get_id(x):
        v = safe_text(x.get(key))
        return v if v else str(hash(str(x)))
    items = [x for x in pool if get_id(x) not in used_ids]
    if not items:
        used_ids.clear()
        items = pool[:]
    item = random.choice(items)
    used_ids.add(get_id(item))
    return item

def filter_scenes_by_shoe_type(shoe_type: str):
    f = [s for s in scenes if safe_text(s.get("shoe_type")).lower() == shoe_type.lower()]
    return f if f else scenes

def filter_dialogues(shoe_type: str, tone: str):
    tone_f = [d for d in dialogues if safe_text(d.get("tone")) == tone]
    if not tone_f:
        tone_f = dialogues
    shoe_f = [d for d in tone_f if safe_text(d.get("shoe_type")).lower() == shoe_type.lower()]
    return shoe_f if shoe_f else tone_f

def split_10s_timeline(n: int):
    n = max(2, min(4, int(n)))
    if n == 2:
        return [(0,5),(5,10)]
    if n == 3:
        return [(0,3.3),(3.3,6.7),(6.7,10)]
    return [(0,2.5),(2.5,5),(5,7.5),(7.5,10)]

def pick_n_unique_scenes(shoe_type: str, n: int):
    pool = filter_scenes_by_shoe_type(shoe_type)
    return [pick_unique(pool, st.session_state.used_scene_ids, "id") for _ in range(n)]

# =========================
# DIALOGUE BANK (giữ nguyên tiếng Việt có dấu)
# =========================
TONE_BANK = {
    "Tự tin": {
        "open": [
            "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn.",
            "Mình thích cảm giác bước đi nhìn gọn và có nhịp.",
            "Mình ưu tiên tổng thể sạch, dễ phối và nhìn sáng dáng."
        ],
        "mid": [
            "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định.",
            "Mình thấy form lên chân nhìn gọn, dễ đi suốt ngày.",
            "Cảm giác di chuyển nhẹ nhàng, không bị rối mắt."
        ],
        "close": [
            "Nhìn tổng thể đơn giản nhưng có điểm tinh tế riêng.",
            "Mình thích kiểu càng tối giản càng dễ tạo phong cách.",
            "Với mình, gọn gàng là đủ đẹp rồi."
        ],
    }
}

def get_dialogue_3_sentences(row: dict, tone: str) -> str:
    bank = TONE_BANK["Tự tin"]
    a = random.choice(bank["open"])
    b = random.choice(bank["mid"])
    c = random.choice(bank["close"])
    return normalize_text(f"{a}\n{b}\n{c}")

def get_dialogue_2_sentences(row: dict, tone: str) -> str:
    bank = TONE_BANK["Tự tin"]
    a = random.choice(bank["open"])
    b = random.choice(bank["mid"])
    return normalize_text(f"{a}\n{b}")

# =========================
# PROMPT BUILDER — FINAL SORA SAFE + LOGO SAFE
# =========================
def build_prompt_unified(
    mode: str,
    shoe_type: str,
    shoe_name: str,
    scene_list: List[dict],
    timeline: List[Tuple[float, float]],
    voice_lines: str,
) -> str:

    if mode == "p1":
        title = "VIDEO SETUP — CINEMATIC SHOE EDITION (NO CAMEO)"
        cast_block = (
            "No people on screen\n"
            "No cameo visible\n"
            f"Voice ID: {CAMEO_VOICE_ID}"
        )
    else:
        title = "VIDEO SETUP — CINEMATIC SHOE EDITION (WITH CAMEO)"
        cast_block = (
            "Cameo appears naturally like a phone review video\n"
            f"Cameo and Voice ID: {CAMEO_VOICE_ID}\n"
            "No hard call to action\n"
            "No price\n"
            "No discount\n"
            "No guarantees"
        )

    scene_blocks = []
    for i, sc in enumerate(scene_list, start=1):
        loc = compact_spaces(safe_text(sc.get("location")))
        light = compact_spaces(safe_text(sc.get("lighting")))
        mot = compact_spaces(safe_text(sc.get("motion")))
        wea = compact_spaces(safe_text(sc.get("weather")))
        mood = compact_spaces(safe_text(sc.get("mood")))

        scene_blocks.append(
            f"Scene {i}:\n"
            f"{loc}\n"
            f"{light}\n"
            f"{mot}\n"
            f"{wea}\n"
            f"Mood {mood}"
        )

    scene_block = "\n\n".join(scene_blocks)

    shoe_name = compact_spaces(shoe_name)
    shoe_type = compact_spaces(shoe_type)
    voice_lines = normalize_text(voice_lines)

    prompt_text = f"""
@phuongnghi18091991

══════════════════════════════════

{title}

══════════════════════════════════
Video dọc 9:16 — 10s
Ultra Sharp PRO 4K output
Realistic cinematic video (NOT static image)
Use the EXACT uploaded shoe image
TikTok-safe absolute

NO text
NO logo overlay
NO watermark
NO blur
NO haze
NO glow

══════════════════════════════════

CAST RULE

══════════════════════════════════
{cast_block}

══════════════════════════════════

SHOE REFERENCE LOCK — ABSOLUTE

══════════════════════════════════
Use only the uploaded shoe image as reference
Keep 100 percent shoe identity
No redesign
No deformation
No guessing
No color shift

══════════════════════════════════

TEXT & LOGO ORIENTATION LOCK — ABSOLUTE

══════════════════════════════════
If the uploaded shoe image contains ANY text or logo

Text orientation MUST be correct
NOT mirrored
NOT reversed
NOT flipped

Camera rotation MUST NOT reverse any logo or text

STRICTLY FORBIDDEN
Mirrored letters
Reversed logos
Flipped symbols

If any angle risks flipping text
Prioritize correct text orientation over camera style

══════════════════════════════════

PRODUCT

══════════════════════════════════
shoe_name: {shoe_name}
shoe_type: {shoe_type}

══════════════════════════════════

SCENE FLOW — AUTO MULTI SHOT

══════════════════════════════════
{scene_block}

══════════════════════════════════

AUDIO MASTERING — CALM STYLE

══════════════════════════════════
0.0–1.2s: no voice
1.2–6.9s: voice on
6.9–10.0s: music only

══════════════════════════════════

VOICEOVER

══════════════════════════════════
{voice_lines}

══════════════════════════════════

HARD RULES — ABSOLUTE

══════════════════════════════════
NO text overlay
NO mirrored logo
NO reversed letters
NO shoe distortion
NO incorrect shoe
"""

    return normalize_text(prompt_text).strip()

# =========================
# UI
# =========================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
    mode = st.radio("Prompt mode", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone = st.selectbox("Tone", ["Tự tin"], index=0)
    scene_count = st.slider("Shots inside total 10s", 2, 4, 4)
    count = st.slider("Prompts per click", 1, 10, 3)

with right:
    st.info("Total video duration always 10 seconds")
    st.info("Prompt format matches old working Sora style")
    st.info("Text & Logo lock enabled")

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

    shoe_type = st.selectbox("shoe_type", SHOE_TYPES, index=2)

    if st.button("Generate Prompt", use_container_width=True):
        arr = []
        for _ in range(count):
            d_pool = filter_dialogues(shoe_type, "Tự tin")
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)

            if mode.startswith("PROMPT 1"):
                voice_lines = get_dialogue_3_sentences(d, tone)
                p = build_prompt_unified("p1", shoe_type, shoe_name, scene_list, timeline, voice_lines)
            else:
                voice_2 = get_dialogue_2_sentences(d, tone)
                disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Nội dung chỉ mang tính chia sẻ trải nghiệm."
                voice_lines = normalize_text(f"{voice_2}\n{disclaimer}")
                p = build_prompt_unified("p2", shoe_type, shoe_name, scene_list, timeline, voice_lines)

            arr.append(p)

        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.markdown("Output prompts")
        tabs = st.tabs([str(i + 1) for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=420)
                copy_button_unicode_safe(prompts[i], key=f"copy_{i}")

st.divider()
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.generated_prompts = []
    st.success("Reset done")
