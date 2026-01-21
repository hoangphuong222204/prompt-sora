import textwrap, os, re, json, unicodedata, pathlib

api_key = "AIzaSyC4ls7YKlq4eNf_FoFZnEscionCig0aSuI"

app_code = f'''\
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
# - Unicode copy-safe (no broken Vietnamese when paste)
# - Text/logo orientation lock (no mirrored logo)
# - Hard-coded Gemini API key (as requested)
# ==========================================================

st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 - Total 10s - Multi-shot - Anti-duplicate - TikTok Shop SAFE - Copy Safe Unicode - Style Pack")

# --------------------------
# HARD-CODED KEY (USER REQUEST)
# --------------------------
HARD_CODE_API_KEY = "{api_key}"

CAMEO_VOICE_ID = "@phuongnghi18091991"

SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]
OPTIONAL_FILES = ["voice_style_library.csv"]  # optional: to diversify narration delivery style

# =========================
# TEXT NORMALIZE (COPY SAFE)
# =========================
ZERO_WIDTH_PATTERN = r"[\\u200b\\u200c\\u200d\\uFEFF]"

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # Normalize combining marks to composed form
    try:
        s = unicodedata.normalize("NFC", s)
    except Exception:
        pass
    # Remove zero-width / BOM
    s = re.sub(ZERO_WIDTH_PATTERN, "", s)
    # Normalize newlines
    s = s.replace("\\r\\n", "\\n").replace("\\r", "\\n")
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
    s = re.sub(r"[ \\t]+", " ", s).strip()
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
    # keep it short, sora-friendly
    if len(s) > 140:
        s = s[:140].rstrip() + "."
    return normalize_text(s)

# =========================
# COPY BUTTON (UNICODE SAFE)
# =========================
def copy_button_unicode_safe(text: str, key: str):
    """
    Copy via navigator.clipboard.writeText using JSON string payload (keeps Unicode safely).
    Avoid base64/atob issues and avoids hidden characters via normalize_text.
    """
    text = normalize_text(text)
    payload = json.dumps(text)  # safe JS string
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
    st.error("Missing files: " + ", ".join(missing) + " (must be in same folder as app.py)")
    st.stop()

# =========================
# LOAD CSV (UTF-8 friendly)
# =========================
@st.cache_data
def read_csv_flexible(path: str) -> pd.DataFrame:
    # Try utf-8-sig first to handle BOM, then utf-8
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
    # take best available column
    for c in ["disclaimer", "text", "mien_tru", "miễn_trừ", "note", "content", "noi_dung", "line", "script"]:
        if c in df.columns:
            arr = df[c].dropna().astype(str).tolist()
            return [normalize_text(x) for x in arr if normalize_text(x)]
    # fallback last column
    col = df.columns[-1]
    arr = df[col].dropna().astype(str).tolist()
    return [normalize_text(x) for x in arr if normalize_text(x)]

@st.cache_data
def load_voice_styles_optional() -> List[dict]:
    p = Path("voice_style_library.csv")
    if not p.exists():
        return []
    df = read_csv_flexible(str(p))
    df.columns = [c.strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records")

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimers()
voice_styles_csv = load_voice_styles_optional()

# =========================
# SESSION – ANTI DUP
# =========================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "used_style_ids" not in st.session_state:
    st.session_state.used_style_ids = set()
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
    """
    Total duration ALWAYS 10.0 seconds.
    2 shots: 0-5, 5-10
    3 shots: 0-3.3, 3.3-6.7, 6.7-10
    4 shots: 0-2.5, 2.5-5, 5-7.5, 7.5-10
    """
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
    out = []
    for _ in range(n):
        s = pick_unique(pool, st.session_state.used_scene_ids, "id")
        out.append(s)
    return out

# =========================
# SHOE TYPE DETECT (AI optional)
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
    if re.fullmatch(r"[\\d_\\-\\.]+", n):
        return "leather"
    return "sneaker"

def gemini_detect_shoe_type(img: Image.Image) -> Tuple[Optional[str], str]:
    """
    Returns (shoe_type or None, raw_text).
    Uses hard-coded API key and auto-picks an available model to avoid 404 NotFound.
    """
    api_key = (HARD_CODE_API_KEY or "").strip()
    if not api_key:
        return None, "NO_KEY"

    try:
        import google.generativeai as genai
    except Exception as e:
        return None, f"IMPORT_FAIL: {type(e).__name__}"

    try:
        genai.configure(api_key=api_key)

        models = genai.list_models()
        available = []
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append(m.name)
        if not available:
            return None, "NO_MODELS"

        preferred = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro-vision",
        ]
        picked = None
        for p in preferred:
            if p in available:
                picked = p
                break
        if not picked:
            picked = available[0]

        model = genai.GenerativeModel(picked)

        prompt = (
            "You are a shoe classification system.\\n"
            "Return ONLY ONE label from this list:\\n"
            f"{', '.join(SHOE_TYPES)}\\n\\n"
            "Rules:\\n"
            "- Return exactly one word.\\n"
            "- No explanations.\\n"
            "- If dress shoe / loafer / oxford / derby => leather.\\n"
            "- If sports sneaker => sneaker or runner.\\n"
        )

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
# VOICE STYLE PACK (DIVERSIFY READING STYLE)
# =========================
VOICE_STYLE_PACK_FALLBACK = [
    "Calm, slow luxury pacing; warm confident male voice; natural pauses; friendly tone",
    "Bright and upbeat, but controlled; clear articulation; slight smile in voice; not salesy",
    "Low, deep, confident; steady tempo; short phrases; premium showroom vibe",
    "Soft storytelling; relaxed tempo; gentle emphasis on comfort; intimate phone-video feel",
    "Energetic but clean; slightly faster tempo; sporty vibe; keep it natural",
]

def pick_voice_style() -> str:
    # If optional CSV exists, try columns: id, style (or text)
    if voice_styles_csv:
        row = pick_unique(voice_styles_csv, st.session_state.setdefault("used_voice_style_ids", set()), "id")
        for c in ["style", "text", "line", "content", "noi_dung"]:
            v = safe_text(row.get(c))
            if v:
                return v
    return random.choice(VOICE_STYLE_PACK_FALLBACK)

# =========================
# VISUAL STYLE PACK (5 DIFFERENT STYLES PER BATCH)
# =========================
STYLE_PACK = [
    {"id": "style_01", "name": "Bright boutique studio", "lens": "40-50mm", "grade": "clean bright luxury", "exposure": "bright exposure, HDR+, no underexposure", "camera": "smooth orbit and push-in"},
    {"id": "style_02", "name": "Daylight cafe minimal", "lens": "35-45mm", "grade": "soft daylight, crisp edges", "exposure": "daylight exposure, keep highlights controlled", "camera": "handheld-stable phone realism, micro sway"},
    {"id": "style_03", "name": "Modern street morning", "lens": "28-35mm", "grade": "fresh morning contrast", "exposure": "bright, clean blacks, no dark scene", "camera": "tracking low angle, smooth glide"},
    {"id": "style_04", "name": "Penthouse window light", "lens": "50mm", "grade": "premium neutral grade", "exposure": "bright window light, high clarity, no haze", "camera": "slow pan + gentle dolly"},
    {"id": "style_05", "name": "Showroom table macro", "lens": "60-85mm macro", "grade": "ultra sharp product macro", "exposure": "high key lighting, zero blur", "camera": "macro rack but keep shoe sharp"},
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
# DIALOGUE (FROM CSV) + OPTIONAL AI GENERATION
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
    parts = [p.strip() for p in re.split(r"[.!?]+", t) if p.strip()]
    return parts

def get_dialogue_3_sentences(row: dict, tone: str) -> str:
    # Prefer CSV; fallback to generic bank
    candidate = get_dialogue_from_csv(row)
    if candidate:
        parts = split_sentences(candidate)
        if len(parts) >= 3:
            a, b, c = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            a, b = parts[0], parts[1]
            c = "Nhìn tổng thể gọn gàng, mình thấy rất dễ dùng hằng ngày."
        elif len(parts) == 1:
            a = parts[0]
            b = "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định."
            c = "Tổng thể tối giản nhưng nhìn vẫn có điểm tinh tế."
        else:
            a = "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
            b = "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định."
            c = "Nhìn tổng thể tối giản nhưng nhìn vẫn có điểm tinh tế."
    else:
        a = "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
        b = "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định."
        c = "Nhìn tổng thể tối giản nhưng nhìn vẫn có điểm tinh tế."

    a, b, c = ensure_end_punct(a), ensure_end_punct(b), ensure_end_punct(c)
    return normalize_text(f"{a}\\n{b}\\n{c}")

def get_dialogue_2_sentences(row: dict, tone: str) -> str:
    candidate = get_dialogue_from_csv(row)
    if candidate:
        parts = split_sentences(candidate)
        if len(parts) >= 2:
            a, b = parts[0], parts[1]
        elif len(parts) == 1:
            a = parts[0]
            b = "Cảm giác di chuyển khá nhẹ nhàng và dễ chịu."
        else:
            a = "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
            b = "Cảm giác di chuyển khá nhẹ nhàng và dễ chịu."
    else:
        a = "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
        b = "Cảm giác di chuyển khá nhẹ nhàng và dễ chịu."

    a, b = ensure_end_punct(a), ensure_end_punct(b)
    return normalize_text(f"{a}\\n{b}")

def gemini_generate_voice_lines(shoe_type: str, tone: str, n_lines: int = 3) -> Tuple[Optional[str], str]:
    """
    Generate Vietnamese experience-style lines (TikTok-safe, no price/discount/CTA).
    Uses the same hard-coded key. This is NOT free forever; depends on your Google quota/tier.
    Returns (text or None, raw_status).
    """
    api_key = (HARD_CODE_API_KEY or "").strip()
    if not api_key:
        return None, "NO_KEY"

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # use any available text model that supports generateContent
        models = genai.list_models()
        available = []
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append(m.name)
        if not available:
            return None, "NO_MODELS"

        # prefer flash/pro
        preferred = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
        picked = None
        for p in preferred:
            if p in available:
                picked = p
                break
        if not picked:
            picked = available[0]

        model = genai.GenerativeModel(picked)

        # Strong safety constraints
        req = (
            "Bạn là người review giày nam theo kiểu chia sẻ trải nghiệm.\\n"
            "Viết đúng {n} câu NGẮN, tự nhiên, giọng nam.\\n"
            "Không kêu gọi mua, không nói giá, không nói khuyến mãi, không cam kết, không so sánh, không nói vật liệu nhạy cảm.\\n"
            "Chỉ nói cảm giác đi, phối đồ, tổng thể gọn gàng.\\n"
            "Mỗi câu 8-14 từ, có dấu câu đầy đủ.\\n"
            "Chỉ trả về các câu, mỗi câu trên 1 dòng.\\n"
        ).format(n=n_lines)

        ctx = f"shoe_type: {shoe_type}. tone: {tone}."
        resp = model.generate_content(req + "\\n" + ctx)
        text = normalize_text(getattr(resp, "text", "") or "")
        if not text:
            return None, f"{picked} -> EMPTY"

        # enforce line count
        lines = [ensure_end_punct(x) for x in text.split("\\n") if compact_spaces(x)]
        lines = lines[:n_lines]
        if len(lines) < n_lines:
            # pad safe lines
            pads = [
                "Nhìn tổng thể gọn gàng, mình thấy rất dễ phối đồ.",
                "Đi một lúc vẫn thấy nhịp bước khá ổn định.",
                "Cảm giác mang lên chân nhìn sạch và tự nhiên.",
            ]
            while len(lines) < n_lines:
                lines.append(pads[len(lines) % len(pads)])
        return normalize_text("\\n".join(lines)), f"{picked} -> OK"
    except Exception as e:
        return None, f"CALL_FAIL: {type(e).__name__}: {e}"

# =========================
# PROMPT BUILDER (MATCH OLD WORKING STYLE)
# =========================
SEP = "══════════════════════════════════"

def build_prompt_old_style(
    mode: str,
    shoe_type: str,
    shoe_name: str,
    style: dict,
    scene_list: List[dict],
    timeline: List[Tuple[float, float]],
    voice_lines: str,
    voice_style_line: str,
) -> str:
    # mode blocks
    if mode == "p1":
        title = "VIDEO SETUP — SLOW LUXURY EDITION (NO CAMEO) (FINAL • TEXT ORIENTATION & SHARPNESS LOCK)"
        cameo_block = (
            "NO people on screen\\n"
            "NO cameo visible\\n"
            f"VOICE ID: {CAMEO_VOICE_ID}"
        )
        disclaimer_block = ""
    else:
        title = "VIDEO SETUP — SLOW LUXURY EDITION (WITH CAMEO) (FINAL • TEXT ORIENTATION & SHARPNESS LOCK)"
        cameo_block = (
            "Cameo appears naturally like a phone review video\\n"
            "Cameo is stable, not over-acting\\n"
            f"CAMEO & VOICE ID: {CAMEO_VOICE_ID}\\n"
            "No hard call to action, no price, no discount, no guarantees"
        )
        disclaimer_block = ""

    # scenes (keep simple but clear)
    shot_lines = []
    for idx, (sc, (a, b)) in enumerate(zip(scene_list, timeline), start=1):
        loc = compact_spaces(safe_text(sc.get("location")))
        light = compact_spaces(safe_text(sc.get("lighting")))
        mot = compact_spaces(safe_text(sc.get("motion")))
        wea = compact_spaces(safe_text(sc.get("weather")))
        mood = compact_spaces(safe_text(sc.get("mood")))
        shot_lines.append(
            f"{a:.1f}–{b:.1f}s: {loc}. {light}. Camera {mot}. Weather {wea}. Mood {mood}."
        )
    shot_block = "\\n".join([ensure_end_punct(x) for x in shot_lines])

    shoe_name = compact_spaces(shoe_name)
    shoe_type = compact_spaces(shoe_type)
    voice_lines = normalize_text(voice_lines)
    voice_style_line = compact_spaces(voice_style_line)

    prompt = f"""
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
{cameo_block}

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
shoe_name: {shoe_name}
shoe_type: {shoe_type}

{SEP}

SHOT LIST — TOTAL 10s (MULTI-SHOT)

{SEP}
{shot_block}

{SEP}

AUDIO MASTERING — CALM & CLEAR

{SEP}
Voice style: {voice_style_line}
0.0–1.2s: NO voice, light ambient only
1.2–6.9s: VOICE ON
6.9–10.0s: VOICE OFF completely, music only, gentle fade-out

{SEP}

VOICEOVER (1.2–6.9s)

{SEP}
{voice_lines}

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

    return normalize_text(prompt)

# =========================
# UI
# =========================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
    mode = st.radio("Prompt mode", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone = st.selectbox("Tone", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)

    # 4 shots default (user asked 4 shots)
    scene_count = st.slider("Shots inside total 10s", 2, 4, 4)

    # 5 prompts default
    count = st.slider("Prompts per click", 1, 10, 5)

    # voice source
    voice_source = st.radio("Voice lines source", ["From CSV (anti-duplicate)", "AI (Gemini) generate"], index=0)

    # shoe_type detect
    detect_mode = st.selectbox("shoe_type detect", ["AI (image) - preferred", "Auto (filename) - fallback", "Manual"], index=0)

with right:
    st.subheader("Notes")
    st.write("Total duration is always 10 seconds. Default is 4 shots.")
    st.write("Each click generates 5 prompts, each prompt has a different visual style.")
    st.write("Prompt 1: 3 voice lines. Prompt 2: 2 voice lines + 1 short disclaimer (3 lines total).")
    st.write("Text and logo orientation lock is ON (no mirrored logo).")
    st.write("Brightness lock is ON to avoid dark and blurry outputs.")

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
        st.success("Manual shoe_type: " + shoe_type)

    st.caption("shoe_name: " + shoe_name)

    if st.button("Generate", use_container_width=True):
        arr = []
        batch_style_used = set()

        for _ in range(count):
            # pick different style each prompt
            style = pick_unique_style_for_batch(batch_style_used)

            # pick scenes
            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)

            # voice style
            voice_style_line = pick_voice_style()

            # voice lines
            if voice_source.startswith("AI"):
                # Prompt 1 -> 3 lines; Prompt 2 -> 2 lines (then disclaimer)
                need = 3 if mode.startswith("PROMPT 1") else 2
                gen, raw = gemini_generate_voice_lines(shoe_type, tone, n_lines=need)
                if gen:
                    voice_lines_core = gen
                else:
                    # fallback to CSV
                    d_pool = filter_dialogues(shoe_type, tone)
                    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
                    voice_lines_core = get_dialogue_3_sentences(d, tone) if need == 3 else get_dialogue_2_sentences(d, tone)
                st.caption("AI voice: " + raw)
            else:
                d_pool = filter_dialogues(shoe_type, tone)
                d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
                voice_lines_core = get_dialogue_3_sentences(d, tone) if mode.startswith("PROMPT 1") else get_dialogue_2_sentences(d, tone)

            # Prompt 2 requires: 2 voice + 1 disclaimer = 3 total lines
            if mode.startswith("PROMPT 2"):
                disc_raw = random.choice(disclaimers_p2) if disclaimers_p2 else "Nội dung chỉ mang tính chia sẻ trải nghiệm."
                disc = short_disclaimer(disc_raw)
                voice_lines = normalize_text(f"{voice_lines_core}\\n{disc}")
            else:
                voice_lines = voice_lines_core

            p = build_prompt_old_style(
                mode="p1" if mode.startswith("PROMPT 1") else "p2",
                shoe_type=shoe_type,
                shoe_name=shoe_name,
                style=style,
                scene_list=scene_list,
                timeline=timeline,
                voice_lines=voice_lines,
                voice_style_line=voice_style_line,
            )
            arr.append(p)

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

# Reset anti-duplicate: clears used ids so prompts can repeat again after exhaustion
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.used_style_ids.clear()
    if "used_voice_style_ids" in st.session_state:
        st.session_state.used_voice_style_ids.clear()
    st.session_state.generated_prompts = []
    st.success("Reset done.")
'''
# write file
path = "/mnt/data/app.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(app_code)

path
