import streamlit as st
import pandas as pd
import random
import re
import json
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

# ==========================================================
# Sora Prompt Studio Pro - Director Edition (SORA ENGINE SAFE)
# - Total 10s prompt
# - 2 modes: Prompt 1 (no cameo) / Prompt 2 (with cameo)
# - 4 shots inside 10s (default), optional 2-4
# - 5 prompts per click (default)
# - 5 different visual styles per batch
# - Unicode copy-safe
# - Text/logo orientation lock (no mirrored logo)
# - Hard-coded Gemini API key (as requested)
# ==========================================================

st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 - Total 10s - Multi-shot - Anti-duplicate - TikTok Shop SAFE - Copy Safe Unicode - Style Pack")

# --------------------------
# HARD-CODED KEY (USER REQUEST)
# --------------------------
HARD_CODE_API_KEY = "AIzaSyC4ls7YKlq4eNf_FoFZnEscionCig0aSuI"

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
    s = normalize_text(str(v).strip())
    if s.lower() == "nan":
        return ""
    return s

def compact_spaces(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def ensure_end_punct(s: str) -> str:
    s = compact_spaces(s)
    if not s:
        return ""
    if s.endswith((".", "!", "?")):
        return s
    return s + "."

def short_disclaimer(raw: str) -> str:
    s = compact_spaces(normalize_text(raw))
    if not s:
        s = "Nội dung chỉ mang tính chia sẻ trải nghiệm."
    s = ensure_end_punct(s)
    if len(s) > 140:
        s = s[:140].rstrip() + "."
    return normalize_text(s)

# =========================
# COPY BUTTON (UNICODE SAFE)
# =========================
def copy_button_unicode_safe(text: str, key: str):
    text = normalize_text(text)
    payload = json.dumps(text)
    html = f"""
    <button id=\"{key}\" style=\"
        padding:8px 14px;border-radius:10px;border:1px solid #ccc;
        cursor:pointer;background:#fff;font-weight:700;\">COPY</button>
    <span id=\"{key}_s\" style=\"margin-left:8px;font-size:12px;\"></span>
    <script>
    (function() {{
        const btn = document.getElementById(\"{key}\");
        const s = document.getElementById(\"{key}_s\");
        const text = {payload};
        btn.onclick = async () => {{
            try {{
                await navigator.clipboard.writeText(text);
                s.innerText = \"Copied\";
                setTimeout(()=>s.innerText=\"\",1500);
            }} catch(e) {{
                s.innerText = \"Clipboard blocked\";
                setTimeout(()=>s.innerText=\"\",2500);
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
    st.error("Missing files: " + ", ".join(missing) + " (must be in same folder as app.py)")
    st.stop()

# =========================
# LOAD CSV (UTF-8 friendly)
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
def load_disclaimers():
    df = read_csv_flexible("disclaimer_prompt2.csv")
    df.columns = [c.strip() for c in df.columns.tolist()]
    for c in ["disclaimer", "text", "mien_tru", "miễn_trừ", "note", "content", "noi_dung", "line", "script"]:
        if c in df.columns:
            arr = df[c].dropna().astype(str).tolist()
            return [normalize_text(x) for x in arr if normalize_text(x)]
    col = df.columns[-1]
    arr = df[col].dropna().astype(str).tolist()
    return [normalize_text(x) for x in arr if normalize_text(x)]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimers()

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

def split_10s_timeline(n: int) -> List[Tuple[float, float]]:
    n = max(2, min(4, int(n)))
    if n == 2:
        cuts = [0.0, 5.0, 10.0]
    elif n == 3:
        cuts = [0.0, 3.3, 6.7, 10.0]
    else:
        cuts = [0.0, 2.5, 5.0, 7.5, 10.0]
    return [(cuts[i], cuts[i + 1]) for i in range(n)]

def pick_n_unique_scenes(shoe_type: str, n: int) -> List[dict]:
    pool = filter_scenes_by_shoe_type(shoe_type)
    return [pick_unique(pool, st.session_state.used_scene_ids, "id") for _ in range(n)]

# =========================
# SHOE TYPE DETECT
# =========================
def detect_shoe_from_filename(name: str) -> str:
    n = (name or "").lower()
    rules = [
        ("boots",   ["boot", "chelsea", "combat", "martin"]),
        ("sandals", ["sandal", "sandals", "dep", "dép", "slipper", "slides"]),
        ("leather", ["loafer", "loafers", "moc", "moccasin", "horsebit", "oxford", "derby", "tassel", "brogue",
                     "giaytay", "giày tây", "giay_da", "giayda", "dressshoe"]),
        ("runner",  ["runner", "running", "jog", "marathon", "gym", "train", "thethao", "thể thao", "sport"]),
        ("casual",  ["casual", "daily", "everyday", "basic"]),
        ("luxury",  ["lux", "premium", "quietlux", "quiet_lux", "highend", "boutique"]),
        ("sneaker", ["sneaker", "sneakers", "kicks", "street"])
    ]
    for shoe_type, keys in rules:
        if any(k in n for k in keys):
            return shoe_type
    if re.fullmatch(r"[\d_\-\.]+", n):
        return "leather"
    return "sneaker"

def gemini_detect_shoe_type(img: Image.Image) -> Tuple[Optional[str], str]:
    api_key = (HARD_CODE_API_KEY or "").strip()
    if not api_key:
        return None, "NO_KEY"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        models = genai.list_models()
        available = [m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
        if not available:
            return None, "NO_MODELS"
        preferred = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro-vision"]
        picked = next((p for p in preferred if p in available), available[0])
        model = genai.GenerativeModel(picked)
        prompt = "Return ONLY ONE label from: " + ", ".join(SHOE_TYPES) + ". No explanation."
        resp = model.generate_content([prompt, img])
        text = (getattr(resp, "text", "") or "").strip().lower()
        raw = f"{picked} -> {text}" if text else f"{picked} -> EMPTY_TEXT"
        norm = re.sub(r"[^a-z_]", "", text)
        if norm in SHOE_TYPES:
            return norm, raw
        for t in SHOE_TYPES:
            if t in norm:
                return t, raw
        return None, raw
    except Exception as e:
        return None, f"CALL_FAIL: {type(e).__name__}: {e}"

# =========================
# VOICE STYLE PACK
# =========================
VOICE_STYLE_PACK = [
    "Calm, slow luxury pacing; warm confident male voice; natural pauses; friendly tone",
    "Bright and upbeat but controlled; clear articulation; slight smile in voice; not salesy",
    "Low, deep, confident; steady tempo; short phrases; premium showroom vibe",
    "Soft storytelling; relaxed tempo; gentle emphasis; intimate phone-video feel",
    "Energetic but clean; slightly faster tempo; sporty vibe; keep it natural",
]

def pick_voice_style() -> str:
    return random.choice(VOICE_STYLE_PACK)

# =========================
# VISUAL STYLE PACK (5 DIFFERENT STYLES PER BATCH)
# =========================
STYLE_PACK = [
    {"id":"style_01","name":"Bright boutique studio","lens":"40-50mm","grade":"clean bright luxury","exposure":"bright exposure, HDR+, no underexposure","camera":"smooth orbit and push-in"},
    {"id":"style_02","name":"Daylight cafe minimal","lens":"35-45mm","grade":"soft daylight, crisp edges","exposure":"daylight exposure, highlights controlled","camera":"handheld-stable phone realism, micro sway"},
    {"id":"style_03","name":"Modern street morning","lens":"28-35mm","grade":"fresh morning contrast","exposure":"bright, clean blacks, no dark scene","camera":"tracking low angle, smooth glide"},
    {"id":"style_04","name":"Penthouse window light","lens":"50mm","grade":"premium neutral grade","exposure":"bright window light, high clarity, no haze","camera":"slow pan + gentle dolly"},
    {"id":"style_05","name":"Showroom table macro","lens":"60-85mm macro","grade":"ultra sharp product macro","exposure":"high key lighting, zero blur","camera":"macro movement but keep shoe sharp"},
]

def pick_unique_style_for_batch(batch_used: set) -> dict:
    pool = [s for s in STYLE_PACK if s["id"] not in batch_used]
    if not pool:
        batch_used.clear()
        pool = STYLE_PACK[:]
    s = random.choice(pool)
    batch_used.add(s["id"])
    return s

# =========================
# DIALOGUE
# =========================
def get_dialogue_from_csv(row: dict) -> str:
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row and safe_text(row.get(col)):
            return safe_text(row.get(col))
    return ""

def split_sentences(text: str) -> List[str]:
    t = safe_text(text)
    if not t:
        return []
    return [p.strip() for p in re.split(r"[.!?]+", t) if p.strip()]

def get_dialogue_3_sentences(row: dict, tone: str) -> str:
    parts = split_sentences(get_dialogue_from_csv(row))
    a = parts[0] if len(parts) > 0 else "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
    b = parts[1] if len(parts) > 1 else "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định."
    c = parts[2] if len(parts) > 2 else "Nhìn tổng thể tối giản nhưng vẫn có điểm tinh tế."
    return normalize_text(f"{ensure_end_punct(a)}\n{ensure_end_punct(b)}\n{ensure_end_punct(c)}")

def get_dialogue_2_sentences(row: dict, tone: str) -> str:
    parts = split_sentences(get_dialogue_from_csv(row))
    a = parts[0] if len(parts) > 0 else "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
    b = parts[1] if len(parts) > 1 else "Cảm giác di chuyển khá nhẹ nhàng và dễ chịu."
    return normalize_text(f"{ensure_end_punct(a)}\n{ensure_end_punct(b)}")

# =========================
# PROMPT BUILDER (OLD WORKING FORMAT)
# =========================
SEP = "══════════════════════════════════"

def build_prompt(mode: str, shoe_type: str, shoe_name: str, style: dict, scene_list: List[dict], timeline: List[Tuple[float,float]], voice_lines: str, voice_style_line: str) -> str:
    if mode == "p1":
        title = "VIDEO SETUP — SLOW LUXURY EDITION (NO CAMEO) (FINAL • TEXT ORIENTATION & SHARPNESS LOCK)"
        cast_block = "NO people on screen\nNO cameo visible\nVOICE ID: " + CAMEO_VOICE_ID
    else:
        title = "VIDEO SETUP — SLOW LUXURY EDITION (WITH CAMEO) (FINAL • TEXT ORIENTATION & SHARPNESS LOCK)"
        cast_block = "Cameo appears naturally like a phone review video\nCameo is stable, not over-acting\nCAMEO & VOICE ID: " + CAMEO_VOICE_ID + "\nNo hard call to action, no price, no discount, no guarantees"

    shot_lines = []
    for sc, (a,b) in zip(scene_list, timeline):
        loc = compact_spaces(safe_text(sc.get("location")))
        light = compact_spaces(safe_text(sc.get("lighting")))
        mot = compact_spaces(safe_text(sc.get("motion")))
        wea = compact_spaces(safe_text(sc.get("weather")))
        mood = compact_spaces(safe_text(sc.get("mood")))
        shot_lines.append(f"{a:.1f}–{b:.1f}s: {loc}. {light}. Camera {mot}. Weather {wea}. Mood {mood}.")
    shot_block = "\n".join([ensure_end_punct(x) for x in shot_lines])

    prompt_text = f"""
{CAMEO_VOICE_ID}

{SEP}

{title}

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
MANDATORY
Bright exposure, HDR+, clean blacks, no underexposure
Shoe is the sharpest object on screen in all frames
Zero motion blur on shoe
If movement risks blur, reduce movement
No foggy lighting, no darkness, no noisy shadows

{SEP}

VISUAL STYLE PACK — THIS PROMPT

{SEP}
Style: {style.get("name")}
Lens: {style.get("lens")}
Color grade: {style.get("grade")}
Exposure: {style.get("exposure")}
Camera feel: {style.get("camera")}

{SEP}

CAST RULE

{SEP}
{cast_block}

{SEP}

SHOE REFERENCE — ABSOLUTE LOCK

{SEP}
Use ONLY the uploaded shoe image as reference
LOCK 100 percent shoe identity
Toe shape, panels, stitching, sole, proportions
NO redesign
NO deformation
NO guessing
NO color shift
LACE RULE
If the uploaded shoe image shows laces then keep laces in ALL frames
If the uploaded shoe image shows no laces then ABSOLUTELY NO laces

{SEP}

TEXT & LOGO ORIENTATION LOCK — ABSOLUTE

{SEP}
If the uploaded shoe image contains any text, logo, symbol, number
Text orientation MUST be correct
NOT mirrored
NOT reversed
NOT flipped
Camera orbit and reflections MUST NOT reverse any logo or text
STRICTLY FORBIDDEN
Mirrored letters
Reversed logos
Flipped symbols
If any angle risks flipping text, prioritize correct text orientation over camera style

{SEP}

PRODUCT

{SEP}
shoe_name: {compact_spaces(shoe_name)}
shoe_type: {compact_spaces(shoe_type)}

{SEP}

SHOT LIST — TOTAL 10s (MULTI-SHOT)

{SEP}
{shot_block}

{SEP}

AUDIO MASTERING — CALM & CLEAR

{SEP}
Voice style: {compact_spaces(voice_style_line)}
0.0–1.2s: NO voice, light ambient only
1.2–6.9s: VOICE ON
6.9–10.0s: VOICE OFF completely, music only, gentle fade-out

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
NO shoe distortion
NO incorrect shoe
""".strip()

    return normalize_text(prompt_text)

# =========================
# UI
# =========================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
    mode_ui = st.radio("Prompt mode", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone = st.selectbox("Tone", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    scene_count = st.slider("Shots inside total 10s", 2, 4, 4)
    count = st.slider("Prompts per click", 1, 10, 5)
    detect_mode = st.selectbox("shoe_type detect", ["AI (image) - preferred", "Auto (filename) - fallback", "Manual"], index=0)

with right:
    st.subheader("Notes")
    st.write("Mỗi lần bấm Generate sẽ ra 5 prompt và 5 phong cách video khác nhau.")
    st.write("Có khóa sáng + nét để tránh video tối và mờ.")
    st.write("Có khóa hướng chữ/logo để tránh bị ngược chữ trên giày.")

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

    detected_filename = detect_shoe_from_filename(uploaded.name)

    if detect_mode.startswith("AI"):
        detected_ai, raw_ai = gemini_detect_shoe_type(img)
        if detected_ai:
            shoe_type = detected_ai
            st.success(f"AI shoe_type: {shoe_type}")
            st.caption("AI raw: " + raw_ai)
        else:
            shoe_type = detected_filename
            st.warning("AI shoe_type failed. Using filename fallback.")
            st.caption("AI raw: " + raw_ai)
            st.info("Fallback shoe_type: " + detected_filename)
    elif detect_mode.startswith("Auto"):
        shoe_type = detected_filename
        st.info("Filename shoe_type: " + shoe_type)
    else:
        shoe_type = st.selectbox("Manual shoe_type", SHOE_TYPES, index=SHOE_TYPES.index("leather") if "leather" in SHOE_TYPES else 0)

    if st.button("Generate", use_container_width=True):
        arr = []
        batch_used = set()
        for _ in range(count):
            style = pick_unique_style_for_batch(batch_used)
            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)

            d_pool = filter_dialogues(shoe_type, tone)
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

            if mode_ui.startswith("PROMPT 1"):
                voice_lines = get_dialogue_3_sentences(d, tone)
                mode = "p1"
            else:
                voice_2 = get_dialogue_2_sentences(d, tone)
                disc_raw = random.choice(disclaimers_p2) if disclaimers_p2 else "Nội dung chỉ mang tính chia sẻ trải nghiệm."
                voice_lines = normalize_text(f"{voice_2}\n{short_disclaimer(disc_raw)}")
                mode = "p2"

            prompt = build_prompt(mode, shoe_type, shoe_name, style, scene_list, timeline, voice_lines, pick_voice_style())
            arr.append(prompt)

        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.markdown("Output prompts")
        tabs = st.tabs([str(i + 1) for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=520, key=f"view_{i}")
                copy_button_unicode_safe(prompts[i], key=f"copy_{i}")
else:
    st.warning("Upload a shoe image to begin.")

st.divider()
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.generated_prompts = []
    st.success("Reset done.")
