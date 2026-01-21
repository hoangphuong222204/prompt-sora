import streamlit as st
import pandas as pd
import random
import re
import json
import unicodedata
from pathlib import Path
from typing import List, Tuple
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 - Total 10s - Multi-shot - Anti-duplicate - TikTok Shop SAFE - ASCII-safe prompt")

CAMEO_VOICE_ID = "@phuongnghi18091991"

SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]
REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

# ============================================================
# TEXT NORMALIZE (COPY SAFE)
# - Keep Vietnamese diacritics (Unicode), but remove hidden chars.
# - Force simple ASCII separators in prompt to avoid Sora parser issues.
# ============================================================
ZERO_WIDTH_PATTERN = r"[\u200b\u200c\u200d\uFEFF]"
SMART_PUNCT_PATTERN = r"[\u2018\u2019\u201C\u201D\u2013\u2014]"  # quotes/dashes

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    try:
        s = unicodedata.normalize("NFC", s)
    except Exception:
        pass
    s = re.sub(ZERO_WIDTH_PATTERN, "", s)
    # Replace smart punctuation with plain ASCII
    s = re.sub(SMART_PUNCT_PATTERN, lambda m: {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"', "\u2013": "-", "\u2014": "-"
    }.get(m.group(0), ""), s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()

def compact_spaces(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def safe_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = normalize_text(v)
    if s.lower() == "nan":
        return ""
    return s

def ensure_end_punct(s: str) -> str:
    s = compact_spaces(s)
    if not s:
        return ""
    if s.endswith((".", "!", "?")):
        return s
    return s + "."

def short_disclaimer(raw: str) -> str:
    s = compact_spaces(raw)
    if not s:
        s = "Nội dung chỉ mang tính chia sẻ trải nghiệm."
    s = ensure_end_punct(s)
    if len(s) > 160:
        s = s[:160].rstrip() + "."
    return normalize_text(s)

# ============================================================
# COPY BUTTON (UNICODE SAFE) - no base64/atob
# ============================================================
def copy_button_unicode_safe(text: str, key: str):
    text = normalize_text(text)
    payload = json.dumps(text)  # safe JS string, preserves Unicode
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

# ============================================================
# FILE CHECK
# ============================================================
missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing) + " (must be in same folder as app.py)")
    st.stop()

# ============================================================
# LOAD CSV
# ============================================================
@st.cache_data
def read_csv_flexible(path: str) -> pd.DataFrame:
    # Try utf-8-sig (handles BOM), then utf-8
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
    # Prefer a known column name; else use last column
    cols = [c.strip() for c in df.columns.tolist()]
    df.columns = cols
    pick_col = "disclaimer" if "disclaimer" in cols else cols[-1]
    arr = df[pick_col].dropna().astype(str).tolist()
    return [normalize_text(x) for x in arr if normalize_text(x)]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2()

# ============================================================
# SESSION - ANTI DUP
# ============================================================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []

# ============================================================
# UTILS
# ============================================================
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

def split_10s_timeline(n: int) -> List[Tuple[float, float]]:
    n = max(2, min(4, int(n)))
    if n == 2:
        return [(0.0, 5.0), (5.0, 10.0)]
    if n == 3:
        return [(0.0, 3.3), (3.3, 6.7), (6.7, 10.0)]
    return [(0.0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 10.0)]

def pick_n_unique_scenes(shoe_type: str, n: int):
    pool = filter_scenes_by_shoe_type(shoe_type)
    return [pick_unique(pool, st.session_state.used_scene_ids, "id") for _ in range(n)]

# ============================================================
# DIALOGUE (Vietnamese - keep diacritics)
# NOTE: You can expand this bank later; keeping it stable now.
# ============================================================
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
    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])
    a = ensure_end_punct(random.choice(bank["open"]))
    b = ensure_end_punct(random.choice(bank["mid"]))
    c = ensure_end_punct(random.choice(bank["close"]))
    return normalize_text(f"{a}\n{b}\n{c}")

def get_dialogue_2_sentences(row: dict, tone: str) -> str:
    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])
    a = ensure_end_punct(random.choice(bank["open"]))
    b = ensure_end_punct(random.choice(bank["mid"]))
    return normalize_text(f"{a}\n{b}")

# ============================================================
# PROMPT BUILDER - SORA FRIENDLY
# Key changes vs your last version:
# - ASCII separators only (no box drawing characters)
# - Avoid unusual bullets/symbols
# - Add low-light sharpness enforcement to keep "tối mà vẫn nét"
# - Add text/logo orientation lock (no mirrored / no reversed)
# - Keep layout similar to your old working prompt blocks
# ============================================================
def build_prompt_unified(
    mode: str,
    shoe_type: str,
    shoe_name: str,
    scene_list: List[dict],
    timeline: List[Tuple[float, float]],
    voice_lines: str,
) -> str:
    shoe_name = compact_spaces(shoe_name)
    shoe_type = compact_spaces(shoe_type)
    voice_lines = normalize_text(voice_lines)

    sep = "========================================"

    if mode == "p1":
        title = "VIDEO SETUP - SLOW LUXURY EDITION (NO CAMEO)"
        cast_block = (
            "No people on screen\n"
            "No cameo visible\n"
            f"Voice ID: {CAMEO_VOICE_ID}"
        )
    else:
        title = "VIDEO SETUP - SLOW LUXURY EDITION (WITH CAMEO)"
        cast_block = (
            "Cameo appears naturally like a phone review video\n"
            f"Cameo and Voice ID: {CAMEO_VOICE_ID}\n"
            "No hard call to action\n"
            "No price\n"
            "No discount\n"
            "No guarantees"
        )

    # Scene flow (include times so Sora understands rhythm)
    scene_lines = []
    for idx, (sc, (a, b)) in enumerate(zip(scene_list, timeline), start=1):
        loc = compact_spaces(safe_text(sc.get("location")))
        light = compact_spaces(safe_text(sc.get("lighting")))
        mot = compact_spaces(safe_text(sc.get("motion")))
        wea = compact_spaces(safe_text(sc.get("weather")))
        mood = compact_spaces(safe_text(sc.get("mood")))

        scene_lines.append(
            f"SHOT {idx} {a:.1f}s-{b:.1f}s\n"
            f"Location: {loc}\n"
            f"Lighting: {light}\n"
            f"Camera motion: {mot}\n"
            f"Weather: {wea}\n"
            f"Mood: {mood}\n"
            "Framing: shoe-first closeups, clean background, phone-video realism\n"
            "Camera: stabilize, no shake, no sudden zoom, no jump cuts"
        )
    scene_block = "\n\n".join(scene_lines)

    prompt_text = f"""
{CAMEO_VOICE_ID}
{sep}
{title}
{sep}
Video vertical 9:16, duration 10 seconds
Ultra sharp 4K output
Realistic cinematic video (NOT a still image)
Use the EXACT uploaded shoe image
TikTok-safe absolute

NO on-screen text
NO logo overlay
NO watermark
NO blur
NO haze
NO glow

{sep}
CAST RULE
{sep}
{cast_block}

{sep}
ULTRA SHARPNESS AND LOW LIGHT LOCK
{sep}
Shoe must remain the sharpest object in all frames
If lighting is dark, increase exposure slightly but keep clean blacks
Reduce motion if it causes blur
No motion blur on the shoe
No soft focus, no AI smoothing, no noise
Prefer bright clean lighting even in night scenes (HDR clean, no neon)

{sep}
SHOE REFERENCE LOCK - ABSOLUTE
{sep}
Use only the uploaded shoe image as reference
Keep 100 percent shoe identity: toe shape, panels, stitching, sole, proportions
No redesign, no deformation, no guessing, no color shift
Lace rule: if the reference shoe has laces then keep laces in all frames; if no laces then absolutely no laces

{sep}
TEXT AND LOGO ORIENTATION LOCK - ABSOLUTE
{sep}
If the uploaded shoe image contains any text, letters, numbers, or logo
Text orientation must be correct, not mirrored, not reversed, not flipped
Avoid mirror reflections and mirror shots
If any camera angle risks flipping text, reduce rotation and prioritize correct orientation

{sep}
PRODUCT
{sep}
shoe_name: {shoe_name}
shoe_type: {shoe_type}

{sep}
SHOT PLAN INSIDE 10s
{sep}
{scene_block}

{sep}
AUDIO TIMELINE
{sep}
0.0-1.2s: no voice, light ambient only
1.2-6.9s: voice on (short natural lines)
6.9-10.0s: voice off, music only, gentle fade out

{sep}
VOICEOVER 1.2-6.9s
{sep}
{voice_lines}

{sep}
HARD RULES - ABSOLUTE
{sep}
NO on-screen text
NO mirrored logo
NO reversed letters
NO incorrect shoe
NO shoe distortion
"""
    return normalize_text(prompt_text).strip()

# ============================================================
# UI
# ============================================================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
    mode = st.radio("Prompt mode", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone = st.selectbox("Tone", ["Tự tin"], index=0)
    scene_count = st.slider("Shots inside total 10s", 2, 4, 4)
    count = st.slider("Prompts per click", 1, 10, 3)

with right:
    st.write("Notes:")
    st.write("- Output prompt uses ASCII separators to avoid Sora input issues.")
    st.write("- Vietnamese voice lines are kept (copy-safe).")
    st.write("- Low light + sharpness lock is added to keep 'dark but sharp'.")
    st.write("- Text/logo orientation lock is added (no mirrored logo).")

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
                disclaimer_raw = random.choice(disclaimers_p2) if disclaimers_p2 else "Nội dung chỉ mang tính chia sẻ trải nghiệm."
                disclaimer = short_disclaimer(disclaimer_raw)
                voice_lines = normalize_text(f"{voice_2}\n{disclaimer}")
                p = build_prompt_unified("p2", shoe_type, shoe_name, scene_list, timeline, voice_lines)

            arr.append(p)

        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.write("Output prompts:")
        tabs = st.tabs([str(i + 1) for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=420, key=f"prompt_{i}")
                copy_button_unicode_safe(prompts[i], key=f"copy_{i}")

else:
    st.warning("Upload a shoe image to begin.")

st.divider()
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.generated_prompts = []
    st.success("Reset done.")
