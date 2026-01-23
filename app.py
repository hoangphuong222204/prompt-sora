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
# Sora Prompt Studio Pro — Director Edition (FAST & STABLE)
# - Total 10s prompt
# - Prompt 1: NO cameo (3 câu thoại)
# - Prompt 2: WITH cameo from 0.0s (2 câu quảng cáo nhẹ + 1 câu miễn trừ)
# - 2–4 shots inside 10s (default 4)
# - 5 prompts per click (default)
# - 5 different visual styles per batch (style pack)
# - Unicode copy-safe (không lỗi dấu)
# - Text/logo orientation lock (NO mirrored / reversed / flipped text)
# - Gemini API key nhập trên UI (optional):
#     + AI detect shoe_type
#     + AI generate voice lines (OFF by default to keep it FAST)
# - Strong anti-duplicate (strict sentence signature):
#     + Prompt 1: câu thoại không trùng (không đảo lộn)
#     + Prompt 2: không được trùng bất kỳ câu nào đã dùng ở Prompt 1 (và Prompt 2 trước đó) trong session
# ==========================================================

st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 • Total 10s • Multi-shot • Anti-duplicate • TikTok Shop SAFE • Copy Safe Unicode • FAST Mode")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]
REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

# =========================
# TEXT NORMALIZE (COPY SAFE)
# =========================
ZERO_WIDTH_PATTERN = r"[\u200b\u200c\u200d\uFEFF]"
CONTROL_CHARS_PATTERN = r"[\x00-\x08\x0b\x0c\x0e-\x1f]"

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    try:
        s = unicodedata.normalize("NFC", s)
    except Exception:
        pass
    s = re.sub(ZERO_WIDTH_PATTERN, "", s)
    s = re.sub(CONTROL_CHARS_PATTERN, "", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join([line.rstrip() for line in s.split("\n")])
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
    s = normalize_text(str(v).strip())
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
    s = compact_spaces(normalize_text(raw))
    if not s:
        s = "Nội dung chỉ mang tính chia sẻ trải nghiệm."
    s = ensure_end_punct(s)
    if len(s) > 140:
        s = s[:140].rstrip() + "."
    return normalize_text(s)

def sig_sentence(s: str) -> str:
    # Strict signature: no reordering allowed; normalize case/space/punct only.
    s = compact_spaces(s).lower()
    s = re.sub(r"[“”\"']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
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
    st.components.v1.html(html, height=44)

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
# SESSION
# =========================
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()

# Sentence-level anti-duplicate (strict)
if "used_sentence_sigs" not in st.session_state:
    st.session_state.used_sentence_sigs = set()

# For prompt outputs
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []

# Gemini key
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

# =========================
# UTILS — SCENES
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
# SHOE TYPE DETECT (FAST default: filename)
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
        ("sneaker", ["sneaker", "sneakers", "kicks", "street"]),
    ]
    for shoe_type, keys in rules:
        if any(k in n for k in keys):
            return shoe_type
    if re.fullmatch(r"[\d_\-\.]+", n):
        return "leather"
    return "sneaker"

def gemini_pick_model_name(genai) -> Optional[str]:
    try:
        models = genai.list_models()
        available = []
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append(m.name)
        if not available:
            return None
        preferred = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro-vision",
            "models/gemini-pro",
        ]
        for p in preferred:
            if p in available:
                return p
        return available[0]
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def gemini_detect_shoe_type_cached(img_bytes: bytes, api_key: str) -> Tuple[Optional[str], str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "NO_KEY"
    try:
        import google.generativeai as genai
    except Exception as e:
        return None, f"IMPORT_FAIL: {type(e).__name__}"
    try:
        genai.configure(api_key=api_key)
        picked = gemini_pick_model_name(genai)
        if not picked:
            return None, "NO_MODELS"
        model = genai.GenerativeModel(picked)
        prompt = "Return ONLY ONE label from: " + ", ".join(SHOE_TYPES) + ". No explanation."
        from PIL import Image as PILImage
        import io
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
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
# VISUAL STYLE PACK (5 styles per batch)
# =========================
STYLE_PACK = [
    {"id":"style_01","name":"Bright boutique studio","lens":"40-50mm","grade":"clean bright luxury",
     "exposure":"bright exposure, HDR+, clean blacks, no underexposure",
     "camera":"smooth orbit and gentle push-in, keep shoe ultra sharp"},
    {"id":"style_02","name":"Daylight cafe minimal","lens":"35-45mm","grade":"soft daylight, crisp edges",
     "exposure":"daylight exposure, highlights controlled, no dim corners",
     "camera":"handheld-stable phone realism, micro sway, no blur"},
    {"id":"style_03","name":"Modern street morning","lens":"28-35mm","grade":"fresh morning contrast",
     "exposure":"bright and clean, no dark shadows, no neon",
     "camera":"tracking low angle, smooth glide, shoe remains sharp"},
    {"id":"style_04","name":"Penthouse window light","lens":"50mm","grade":"premium neutral grade",
     "exposure":"bright window light, high clarity, no haze",
     "camera":"slow pan + gentle dolly, avoid motion blur"},
    {"id":"style_05","name":"Showroom table macro","lens":"60-85mm macro","grade":"ultra sharp product macro",
     "exposure":"high-key lighting, zero blur, zero noise",
     "camera":"macro movement but keep shoe the sharpest object"},
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
# DIALOGUE POOL (FAST, strict anti-duplicate)
# =========================
def row_matches(row: dict, shoe_type: str, tone: str) -> bool:
    stype = safe_text(row.get("shoe_type")).lower()
    ttone = safe_text(row.get("tone"))
    ok_tone = (not ttone) or (ttone == tone)
    ok_type = (not stype) or (stype == shoe_type.lower())
    return ok_tone and ok_type

def extract_sentences_from_row(row: dict) -> List[str]:
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung", "noi dung"]:
        if col in row and safe_text(row.get(col)):
            text = safe_text(row.get(col))
            parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
            return [ensure_end_punct(p) for p in parts if p.strip()]
    return []

@st.cache_data(show_spinner=False)
def build_sentence_pool(shoe_type: str, tone: str) -> List[str]:
    pool: List[str] = []
    for r in dialogues:
        if row_matches(r, shoe_type, tone):
            pool.extend(extract_sentences_from_row(r))
    # de-dup inside pool
    uniq = {}
    for s in pool:
        sig = sig_sentence(s)
        if sig and sig not in uniq:
            uniq[sig] = ensure_end_punct(s)
    return list(uniq.values())

def pick_unique_sentences(pool: List[str], n: int, used_sigs_global: set, used_sigs_local: set) -> Optional[List[str]]:
    candidates = [s for s in pool if sig_sentence(s) not in used_sigs_global and sig_sentence(s) not in used_sigs_local]
    if len(candidates) < n:
        return None
    picked = random.sample(candidates, n)
    for s in picked:
        used_sigs_local.add(sig_sentence(s))
    return [ensure_end_punct(s) for s in picked]

def get_prompt1_voice_lines(shoe_type: str, tone: str) -> str:
    pool = build_sentence_pool(shoe_type, tone)
    if len(pool) < 30:
        pool = build_sentence_pool(shoe_type, "Tự tin") or pool
    local = set()
    lines = pick_unique_sentences(pool, 3, st.session_state.used_sentence_sigs, local)
    if not lines:
        st.session_state.used_sentence_sigs.clear()
        local.clear()
        lines = pick_unique_sentences(pool, 3, st.session_state.used_sentence_sigs, local)
    if not lines:
        lines = [
            "Hôm nay mình chọn kiểu gọn gàng để đi cả ngày cho thoải mái.",
            "Bước chân nhìn gọn, nhịp đi đều và rất dễ chịu.",
            "Tổng thể tối giản nhưng lên chân vẫn thấy sang.",
        ]
        local = set(sig_sentence(x) for x in lines)
    for s in local:
        st.session_state.used_sentence_sigs.add(s)
    return normalize_text("\n".join(lines))

def get_prompt2_voice_lines(shoe_type: str, tone: str, disclaimer: str) -> str:
    pool = build_sentence_pool(shoe_type, tone)
    if len(pool) < 30:
        pool = build_sentence_pool(shoe_type, "Tự tin") or pool
    local = set()
    lines = pick_unique_sentences(pool, 2, st.session_state.used_sentence_sigs, local)
    if not lines:
        st.session_state.used_sentence_sigs.clear()
        local.clear()
        lines = pick_unique_sentences(pool, 2, st.session_state.used_sentence_sigs, local)
    if not lines:
        lines = [
            "Mình thích kiểu form gọn, nhìn sạch và dễ phối đồ.",
            "Đi lại nhẹ nhàng, nhịp bước ổn và khá thoải mái.",
        ]
        local = set(sig_sentence(x) for x in lines)

    disc = short_disclaimer(disclaimer)
    disc_sig = sig_sentence(disc)
    if disc_sig in st.session_state.used_sentence_sigs:
        for _ in range(5):
            cand = short_disclaimer(random.choice(disclaimers_p2) if disclaimers_p2 else disc)
            if sig_sentence(cand) not in st.session_state.used_sentence_sigs:
                disc = cand
                disc_sig = sig_sentence(cand)
                break

    for s in local:
        st.session_state.used_sentence_sigs.add(s)
    st.session_state.used_sentence_sigs.add(disc_sig)

    return normalize_text("\n".join([*lines, disc]))

# =========================
# OPTIONAL: Gemini generate voice lines (SLOWER) — cached
# =========================
@st.cache_data(show_spinner=False)
def gemini_generate_voice_lines_cached(api_key: str, shoe_type: str, tone: str, voice_style: str, n_lines: int, seed: int) -> Tuple[Optional[List[str]], str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "NO_KEY"
    try:
        import google.generativeai as genai
    except Exception as e:
        return None, f"IMPORT_FAIL: {type(e).__name__}"
    try:
        genai.configure(api_key=api_key)
        picked = gemini_pick_model_name(genai)
        if not picked:
            return None, "NO_MODELS"
        model = genai.GenerativeModel(picked)
        prompt = f"""
Viết đúng {n_lines} câu tiếng Việt để làm lời thoại review ngắn (video 10 giây) về giày.
Chỉ trả về {n_lines} dòng, mỗi dòng 1 câu, không đánh số, không emoji.
Mỗi câu 8–16 từ.
Tone: {tone}.
Gợi ý cách đọc (giọng nam cameo): {voice_style}.
Nội dung: cảm giác/nhịp bước/độ gọn gàng, tự nhiên như clip điện thoại.
CẤM: giá, giảm giá, khuyến mãi, bảo hành, cam kết tuyệt đối, so sánh hãng khác, công dụng y tế, vật liệu nhạy cảm.
shoe_type: {shoe_type}
random_seed: {seed}
"""
        resp = model.generate_content(prompt)
        text = normalize_text(getattr(resp, "text", "") or "")
        if not text:
            return None, f"{picked} -> EMPTY_TEXT"
        lines = [ensure_end_punct(compact_spaces(x)) for x in text.split("\n") if compact_spaces(x)]
        lines = lines[:n_lines]
        if len(lines) < n_lines:
            return None, f"{picked} -> NOT_ENOUGH_LINES"
        return lines, f"{picked} -> OK"
    except Exception as e:
        return None, f"CALL_FAIL: {type(e).__name__}: {e}"

def ai_or_csv_prompt1(shoe_type: str, tone: str, api_key: str, use_ai_voice: bool, voice_style: str, debug: bool) -> str:
    if use_ai_voice and api_key:
        seed = random.randint(1, 10_000_000)
        lines, dbg = gemini_generate_voice_lines_cached(api_key, shoe_type, tone, voice_style, 3, seed)
        if debug:
            st.caption("AI voice p1: " + dbg)
        if lines:
            sigs = [sig_sentence(x) for x in lines]
            if any(s in st.session_state.used_sentence_sigs for s in sigs):
                return get_prompt1_voice_lines(shoe_type, tone)
            for s in sigs:
                st.session_state.used_sentence_sigs.add(s)
            return normalize_text("\n".join(lines))
    return get_prompt1_voice_lines(shoe_type, tone)

def ai_or_csv_prompt2(shoe_type: str, tone: str, api_key: str, use_ai_voice: bool, voice_style: str, disclaimer: str, debug: bool) -> str:
    if use_ai_voice and api_key:
        seed = random.randint(1, 10_000_000)
        lines, dbg = gemini_generate_voice_lines_cached(api_key, shoe_type, tone, voice_style, 2, seed)
        if debug:
            st.caption("AI voice p2: " + dbg)
        if lines:
            sigs = [sig_sentence(x) for x in lines]
            if any(s in st.session_state.used_sentence_sigs for s in sigs):
                return get_prompt2_voice_lines(shoe_type, tone, disclaimer)
            for s in sigs:
                st.session_state.used_sentence_sigs.add(s)
            disc = short_disclaimer(disclaimer)
            if sig_sentence(disc) in st.session_state.used_sentence_sigs:
                for _ in range(5):
                    cand = short_disclaimer(random.choice(disclaimers_p2) if disclaimers_p2 else disc)
                    if sig_sentence(cand) not in st.session_state.used_sentence_sigs:
                        disc = cand
                        break
            st.session_state.used_sentence_sigs.add(sig_sentence(disc))
            return normalize_text("\n".join([*lines, disc]))
    return get_prompt2_voice_lines(shoe_type, tone, disclaimer)

# =========================
# PROMPT BUILDER
# =========================
SEP = "══════════════════════════════════"

def build_prompt(
    mode: str,
    shoe_type: str,
    shoe_name: str,
    style: dict,
    scene_list: List[dict],
    timeline: List[Tuple[float, float]],
    voice_lines: str,
    voice_style_line: str,
) -> str:

    if mode == "p1":
        title = "VIDEO SETUP — SLOW LUXURY EDITION (NO CAMEO) (FINAL • TEXT ORIENTATION & SHARPNESS LOCK)"
        cast_block = "NO people on screen\nNO cameo visible\nVOICE ID: " + CAMEO_VOICE_ID
        cameo_timeline = "No cameo at any time."
    else:
        title = "VIDEO SETUP — PROMPT 2 (WITH CAMEO FROM START) (FINAL • TEXT ORIENTATION & SHARPNESS LOCK)"
        cast_block = (
            "Cameo appears from 0.0s and stays stable like a phone review video\n"
            "Cameo holds the shoe clearly, not covering details\n"
            "CAMEO & VOICE ID: " + CAMEO_VOICE_ID + "\n"
            "No hard call to action, no price, no discount, no guarantees"
        )
        cameo_timeline = "Cameo visible from 0.0s, face + shoe in frame."

    shot_lines = []
    for sc, (a, b) in zip(scene_list, timeline):
        loc = compact_spaces(safe_text(sc.get("location")))
        light = compact_spaces(safe_text(sc.get("lighting")))
        mot = compact_spaces(safe_text(sc.get("motion")))
        wea = compact_spaces(safe_text(sc.get("weather")))
        mood = compact_spaces(safe_text(sc.get("mood")))
        line = f"{a:.1f}–{b:.1f}s: {loc}. {light}. Camera {mot}. Weather {wea}. Mood {mood}."
        shot_lines.append(ensure_end_punct(line))
    shot_block = "\n".join(shot_lines)

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

NO on-screen text
NO logo overlay
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
CAMEO TIMELINE: {cameo_timeline}

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

CLOSURE ZONE LOCK (lace vs no-lace):
If the uploaded shoe image shows laces / eyelets / lace geometry -> laces MUST exist in EVERY frame
If the uploaded shoe image shows NO laces -> ABSOLUTELY NO laces may appear in any frame

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
If any angle risks flipping text, prioritize correct text orientation over camera style:
reduce rotation, avoid mirror/reflection shots.

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
Voice style (male cameo): {compact_spaces(voice_style_line)}
0.0–1.2s: NO voice, light ambient only
1.2–6.9s: VOICE ON (must finish ALL lines before 6.9s)
6.9–10.0s: VOICE OFF completely, music only, gentle fade-out

{SEP}

VOICEOVER (1.2–6.9s)

{SEP}
{normalize_text(voice_lines)}

{SEP}

HARD RULES — ABSOLUTE

{SEP}
NO on-screen text
NO logo overlay
NO watermark
NO mirrored logo
NO reversed letters
NO shoe distortion
NO incorrect shoe
""".strip()

    return normalize_text(prompt_text)

# =========================
# SIDEBAR: GEMINI KEY + OPTIONS
# =========================
with st.sidebar:
    st.markdown("### Gemini API Key (optional)")
    st.caption("Dán key để AI detect shoe_type / (tuỳ chọn) AI tạo lời thoại. Mặc định OFF để chạy nhanh.")

    key_in = st.text_input("Gemini API key", value=st.session_state.gemini_api_key, type="password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save key", use_container_width=True):
            st.session_state.gemini_api_key = (key_in or "").strip()
            st.success("Saved for this session.")
    with c2:
        if st.button("Clear", use_container_width=True):
            st.session_state.gemini_api_key = ""
            st.info("Cleared.")

    st.markdown("---")
    fast_mode = st.checkbox("FAST mode (recommended for 150 videos/day)", value=True)
    use_ai_detect = st.checkbox("AI detect shoe_type (needs key)", value=False)
    use_ai_voice = st.checkbox("AI generate voice lines (SLOWER, needs key)", value=False)
    show_ai_debug = st.checkbox("Show AI debug", value=False)

# =========================
# UI
# =========================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
    mode_ui = st.radio("Prompt mode", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo (from start)"], index=0)
    tone = st.selectbox("Tone", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    scene_count = st.slider("Shots inside total 10s", 2, 4, 4)
    count = st.slider("Prompts per click", 1, 10, 5)

with right:
    st.subheader("Notes")
    st.write("• 1 click = **5 prompt** + **5 style** khác nhau (mặc định).")
    st.write("• Có khóa **sáng + nét** để hạn chế video tối/mờ.")
    st.write("• Có khóa **hướng chữ/logo** để tránh bị đảo chữ trên giày.")
    st.write("• Anti-duplicate: câu thoại **không trùng** và Prompt 2 **không trùng** bất kỳ câu nào của Prompt 1 (trong session).")
    st.caption("Dialogue cols: " + ", ".join([str(x) for x in dialogue_cols]))
    st.caption("Scene cols: " + ", ".join([str(x) for x in scene_cols]))

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

    detected_filename = detect_shoe_from_filename(uploaded.name)
    shoe_type = detected_filename

    # Determine shoe_type
    if use_ai_detect and st.session_state.gemini_api_key:
        img_bytes = uploaded.getvalue()
        detected_ai, raw_ai = gemini_detect_shoe_type_cached(img_bytes, st.session_state.gemini_api_key)
        if detected_ai:
            shoe_type = detected_ai
            st.success(f"AI shoe_type: {shoe_type}")
            if show_ai_debug:
                st.caption("AI raw: " + raw_ai)
        else:
            shoe_type = detected_filename
            st.warning("AI shoe_type failed. Using filename fallback.")
            if show_ai_debug:
                st.caption("AI raw: " + raw_ai)
            st.info("Fallback shoe_type: " + detected_filename)
    else:
        st.info("shoe_type (filename fallback): " + shoe_type)

    shoe_type = st.selectbox("Confirm / override shoe_type", SHOE_TYPES, index=SHOE_TYPES.index(shoe_type) if shoe_type in SHOE_TYPES else 0)

    if st.button("Generate", use_container_width=True):
        arr = []
        batch_used_styles = set()

        def pick_disclaimer():
            if disclaimers_p2:
                return random.choice(disclaimers_p2)
            return "Nội dung chỉ mang tính chia sẻ trải nghiệm."

        for _ in range(count):
            style = pick_unique_style_for_batch(batch_used_styles)
            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)
            voice_style = pick_voice_style()

            ai_voice_ok = (use_ai_voice and (not fast_mode))

            if mode_ui.startswith("PROMPT 1"):
                mode = "p1"
                voice_lines = ai_or_csv_prompt1(
                    shoe_type=shoe_type,
                    tone=tone,
                    api_key=st.session_state.gemini_api_key,
                    use_ai_voice=ai_voice_ok,
                    voice_style=voice_style,
                    debug=show_ai_debug,
                )
            else:
                mode = "p2"
                voice_lines = ai_or_csv_prompt2(
                    shoe_type=shoe_type,
                    tone=tone,
                    api_key=st.session_state.gemini_api_key,
                    use_ai_voice=ai_voice_ok,
                    voice_style=voice_style,
                    disclaimer=pick_disclaimer(),
                    debug=show_ai_debug,
                )

            prompt = build_prompt(
                mode=mode,
                shoe_type=shoe_type,
                shoe_name=shoe_name,
                style=style,
                scene_list=scene_list,
                timeline=timeline,
                voice_lines=voice_lines,
                voice_style_line=voice_style,
            )
            arr.append(prompt)

        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.markdown("### Output prompts")
        tabs = st.tabs([str(i + 1) for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=640, key=f"view_{i}")
                copy_button_unicode_safe(prompts[i], key=f"copy_{i}")
else:
    st.warning("Upload a shoe image to begin.")

st.divider()
cA, cB = st.columns(2)
with cA:
    if st.button("Reset anti-duplicate (sentences + scenes)"):
        st.session_state.used_sentence_sigs.clear()
        st.session_state.used_scene_ids.clear()
        st.session_state.generated_prompts = []
        st.success("Reset done.")
with cB:
    if st.button("Clear only sentences anti-dup"):
        st.session_state.used_sentence_sigs.clear()
        st.success("Cleared sentence history (scenes kept).")
