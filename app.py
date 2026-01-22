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
# - 2–4 shots inside 10s (default 4)
# - 5 prompts per click (default)
# - 5 different visual styles per batch (style pack)
# - Voice style pack (đa dạng cách đọc)
# - Unicode copy-safe (không lỗi dấu)
# - Text/logo orientation lock (no mirrored/reversed logo)
# - Gemini API key nhập trực tiếp trên UI (không hard-code)
# - Optional: Gemini auto-generate voice lines (text) per prompt
# - HARD ANTI-DUP: no repeated lines across prompts in a batch + across session
#   and ALSO no "word re-order" duplicates (bag-of-words signature)
#
# Prompt rules requested:
# - PROMPT 1: 3 lines voice (all unique; no re-order dup)
# - PROMPT 2: 2 lines voice + 1 disclaimer + 1 extra point (all unique)
#             and MUST NOT duplicate any line used in PROMPT 1 (same batch/session)
# ==========================================================

st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 • Total 10s • Multi-shot • Anti-duplicate HARD • TikTok Shop SAFE • Copy Safe Unicode • Style Pack • Key-in-UI • AI Voice Lines")

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

def sanitize_for_sora(s: str) -> str:
    return normalize_text(s)

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
    text = sanitize_for_sora(text)
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
# SESSION – ANTI DUP + KEY
# =========================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

# HARD anti-dup for voice lines (across the whole session)
if "used_voice_sig_exact" not in st.session_state:
    st.session_state.used_voice_sig_exact = set()
if "used_voice_sig_bow" not in st.session_state:
    st.session_state.used_voice_sig_bow = set()

# =========================
# HARD ANTI-DUP SIGNATURES
# =========================
_KEEP_LETTERS_RE = re.compile(r"[^0-9a-zA-ZÀ-ỹĐđ\s]+", flags=re.UNICODE)

def _tok(s: str) -> List[str]:
    s = sanitize_for_sora(s).lower()
    s = _KEEP_LETTERS_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split(" ")

def sig_exact(s: str) -> str:
    return " ".join(_tok(s))

def sig_bow(s: str) -> str:
    toks = _tok(s)
    toks.sort()
    return " ".join(toks)

def is_line_used(line: str, forbid_exact: set, forbid_bow: set) -> bool:
    ex = sig_exact(line)
    bw = sig_bow(line)
    if not ex or not bw:
        return True
    return (
        ex in forbid_exact
        or bw in forbid_bow
        or ex in st.session_state.used_voice_sig_exact
        or bw in st.session_state.used_voice_sig_bow
    )

def mark_lines_used(lines: List[str], forbid_exact: set, forbid_bow: set):
    for ln in lines:
        ex = sig_exact(ln)
        bw = sig_bow(ln)
        if not ex or not bw:
            continue
        forbid_exact.add(ex)
        forbid_bow.add(bw)
        st.session_state.used_voice_sig_exact.add(ex)
        st.session_state.used_voice_sig_bow.add(bw)

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

def gemini_detect_shoe_type(img: Image.Image, api_key: str) -> Tuple[Optional[str], str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "NO_KEY"
    try:
        import google.generativeai as genai
    except Exception:
        return None, "IMPORT_FAIL"
    try:
        genai.configure(api_key=api_key)
        picked = gemini_pick_model_name(genai)
        if not picked:
            return None, "NO_MODELS"
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
# DIALOGUE (CSV fallback) + optional Gemini generation
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

def csv_candidate_pool_from_row(row: dict) -> List[str]:
    parts = split_sentences(get_dialogue_from_csv(row))
    return [ensure_end_punct(p) for p in parts if compact_spaces(p)]

DEFAULT_VOICE_POOL = [
    "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn.",
    "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định.",
    "Nhìn tổng thể tối giản nhưng vẫn có điểm tinh tế.",
    "Mình thích kiểu form nhìn gọn, bước đi có nhịp và dễ phối đồ.",
    "Cảm giác lên chân khá nhẹ, di chuyển tự nhiên và không bị gò.",
    "Tổng thể sạch, nhìn đứng dáng và hợp nhiều bối cảnh.",
    "Bước chậm cũng đẹp, bước nhanh vẫn gọn và chắc nhịp.",
]

EXTRA_POINT_POOL = [
    "Form nhìn gọn, dễ phối đồ hằng ngày.",
    "Nhịp bước đều, cảm giác di chuyển khá tự nhiên.",
    "Tổng thể sạch và nhìn rất đứng dáng.",
    "Lên chân nhìn cân đối, không bị rối mắt.",
    "Đi dạo nhẹ cũng thấy thoải mái và ổn định.",
    "Chụp góc thấp vẫn giữ được dáng giày rõ ràng.",
    "Di chuyển mượt, nhìn gọn và có nhịp.",
    "Phong cách tối giản nên dễ match nhiều outfit.",
    "Cảm giác bước chân êm và chắc nhịp hơn mình nghĩ.",
    "Nhìn nghiêng vẫn giữ form đẹp và rõ chi tiết.",
]

def gemini_generate_voice_lines(api_key: str, shoe_type: str, tone: str, voice_style: str, n_lines: int) -> Tuple[Optional[List[str]], str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "NO_KEY"
    try:
        import google.generativeai as genai
    except Exception:
        return None, "IMPORT_FAIL"
    try:
        genai.configure(api_key=api_key)
        picked = gemini_pick_model_name(genai)
        if not picked:
            return None, "NO_MODELS"
        model = genai.GenerativeModel(picked)
        prompt = f"""
Viết đúng {n_lines} câu tiếng Việt để làm lời thoại review ngắn (video 10 giây) về giày.
Chỉ trả về {n_lines} dòng, mỗi dòng 1 câu, không đánh số, không emoji.
Mỗi câu 8–16 từ. Tone: {tone}. Gợi ý cách đọc: {voice_style}.
Nội dung: cảm giác/nhịp bước/độ gọn gàng, nói tự nhiên như clip điện thoại.

CẤM: giá, giảm giá, khuyến mãi, bảo hành, cam kết tuyệt đối, so sánh với sản phẩm khác,
công dụng y tế, vật liệu nhạy cảm, kêu gọi mua ngay.

shoe_type: {shoe_type}
"""
        resp = model.generate_content(prompt)
        text = sanitize_for_sora(getattr(resp, "text", "") or "")
        if not text:
            return None, f"{picked} -> EMPTY_TEXT"
        lines = [ensure_end_punct(compact_spaces(x)) for x in text.split("\n") if compact_spaces(x)]
        lines = lines[:n_lines]
        if len(lines) < n_lines:
            return None, f"{picked} -> NOT_ENOUGH_LINES"
        return lines, f"{picked} -> OK"
    except Exception as e:
        return None, f"CALL_FAIL: {type(e).__name__}: {e}"

def pick_unique_lines_from_pool(candidates: List[str], need: int, forbid_exact: set, forbid_bow: set, max_tries: int = 200) -> List[str]:
    out: List[str] = []
    tries = 0
    if not candidates:
        candidates = DEFAULT_VOICE_POOL[:]
    while len(out) < need and tries < max_tries:
        tries += 1
        cand = ensure_end_punct(random.choice(candidates))
        if not cand:
            continue
        if is_line_used(cand, forbid_exact, forbid_bow):
            continue
        if any(sig_bow(cand) == sig_bow(x) or sig_exact(cand) == sig_exact(x) for x in out):
            continue
        out.append(cand)
        mark_lines_used([cand], forbid_exact, forbid_bow)
    return out

def pick_unique_disclaimer(disclaimers: List[str], forbid_exact: set, forbid_bow: set) -> str:
    pool = disclaimers[:] if disclaimers else ["Nội dung chỉ mang tính chia sẻ trải nghiệm."]
    for _ in range(60):
        cand = short_disclaimer(random.choice(pool))
        if not is_line_used(cand, forbid_exact, forbid_bow):
            mark_lines_used([cand], forbid_exact, forbid_bow)
            return cand
    base = "Nội dung chỉ mang tính chia sẻ trải nghiệm cá nhân."
    cand = ensure_end_punct(base)
    if is_line_used(cand, forbid_exact, forbid_bow):
        cand = ensure_end_punct(base + " Mỗi người có cảm nhận khác nhau")
    mark_lines_used([cand], forbid_exact, forbid_bow)
    return cand

def pick_unique_extra_point(forbid_exact: set, forbid_bow: set) -> str:
    pool = EXTRA_POINT_POOL[:]
    for _ in range(80):
        cand = ensure_end_punct(random.choice(pool))
        if not is_line_used(cand, forbid_exact, forbid_bow):
            mark_lines_used([cand], forbid_exact, forbid_bow)
            return cand
    cand = ensure_end_punct("Mình thấy tổng thể gọn gàng và dễ nhìn.")
    mark_lines_used([cand], forbid_exact, forbid_bow)
    return cand

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
    else:
        title = "VIDEO SETUP — SLOW LUXURY EDITION (WITH CAMEO) (FINAL • TEXT ORIENTATION & SHARPNESS LOCK)"
        cast_block = (
            "Cameo appears naturally like a phone review video\n"
            "Cameo is stable, not over-acting\n"
            "CAMEO & VOICE ID: " + CAMEO_VOICE_ID + "\n"
            "No hard call to action, no price, no discount, no guarantees"
        )

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
{sanitize_for_sora(voice_lines)}

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

    return sanitize_for_sora(prompt_text)

# =========================
# SIDEBAR: GEMINI KEY
# =========================
with st.sidebar:
    st.markdown("### Gemini API Key")
    st.caption("Dán key để AI detect shoe_type và/hoặc auto-generate lời thoại. Key chỉ lưu trong session.")

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

    use_ai_voice = st.checkbox("Auto-generate voice lines with Gemini", value=False)
    show_ai_debug = st.checkbox("Show AI debug", value=False)

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
    st.write("Mỗi lần bấm **Generate** sẽ ra **5 prompt** và **5 phong cách video** khác nhau.")
    st.write("Có khóa **sáng + nét** để tránh video tối/mờ.")
    st.write("Có khóa **hướng chữ/logo** để tránh bị ngược chữ trên giày.")
    st.write("**HARD anti-dup**: không trùng câu, kể cả đảo lộn từ (signature bag-of-words).")
    st.caption("Dialogue cols: " + ", ".join([str(x) for x in dialogue_cols]))
    st.caption("Scene cols: " + ", ".join([str(x) for x in scene_cols]))

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

    detected_filename = detect_shoe_from_filename(uploaded.name)

    if detect_mode.startswith("AI"):
        detected_ai, raw_ai = gemini_detect_shoe_type(img, st.session_state.gemini_api_key)
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
    elif detect_mode.startswith("Auto"):
        shoe_type = detected_filename
        st.info("Filename shoe_type: " + shoe_type)
    else:
        shoe_type = st.selectbox("Manual shoe_type", SHOE_TYPES, index=SHOE_TYPES.index("leather") if "leather" in SHOE_TYPES else 0)
        st.success("Manual shoe_type: " + shoe_type)

    if st.button("Generate", use_container_width=True):
        arr = []
        batch_used_styles = set()

        # batch forbid => ensures Prompt 2 never duplicates Prompt 1 in the same click
        batch_forbid_exact = set()
        batch_forbid_bow = set()

        for _ in range(count):
            style = pick_unique_style_for_batch(batch_used_styles)
            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)
            voice_style = pick_voice_style()

            want_p1 = mode_ui.startswith("PROMPT 1")
            mode = "p1" if want_p1 else "p2"

            d_pool = filter_dialogues(shoe_type, tone)
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
            csv_pool = csv_candidate_pool_from_row(d) + DEFAULT_VOICE_POOL

            voice_lines_list: List[str] = []

            if mode == "p1":
                need = 3
                if use_ai_voice and st.session_state.gemini_api_key:
                    ok = False
                    dbg_last = ""
                    for _try in range(3):
                        gen_lines, dbg = gemini_generate_voice_lines(
                            api_key=st.session_state.gemini_api_key,
                            shoe_type=shoe_type,
                            tone=tone,
                            voice_style=voice_style,
                            n_lines=need,
                        )
                        dbg_last = dbg
                        if not gen_lines:
                            continue
                        tmp_ok = True
                        for ln in gen_lines:
                            if is_line_used(ln, batch_forbid_exact, batch_forbid_bow):
                                tmp_ok = False
                                break
                        if tmp_ok:
                            voice_lines_list = [ensure_end_punct(x) for x in gen_lines]
                            mark_lines_used(voice_lines_list, batch_forbid_exact, batch_forbid_bow)
                            ok = True
                            if show_ai_debug:
                                st.caption("AI voice(P1): " + dbg)
                            break
                    if not ok and show_ai_debug and dbg_last:
                        st.caption("AI voice(P1) fallback: " + dbg_last)

                if not voice_lines_list:
                    voice_lines_list = pick_unique_lines_from_pool(csv_pool, need, batch_forbid_exact, batch_forbid_bow)

            else:
                need = 2
                if use_ai_voice and st.session_state.gemini_api_key:
                    ok = False
                    dbg_last = ""
                    for _try in range(3):
                        gen_lines, dbg = gemini_generate_voice_lines(
                            api_key=st.session_state.gemini_api_key,
                            shoe_type=shoe_type,
                            tone=tone,
                            voice_style=voice_style,
                            n_lines=need,
                        )
                        dbg_last = dbg
                        if not gen_lines:
                            continue
                        tmp_ok = True
                        for ln in gen_lines:
                            if is_line_used(ln, batch_forbid_exact, batch_forbid_bow):
                                tmp_ok = False
                                break
                        if tmp_ok:
                            voice_lines_list = [ensure_end_punct(x) for x in gen_lines]
                            mark_lines_used(voice_lines_list, batch_forbid_exact, batch_forbid_bow)
                            ok = True
                            if show_ai_debug:
                                st.caption("AI voice(P2): " + dbg)
                            break
                    if not ok and show_ai_debug and dbg_last:
                        st.caption("AI voice(P2) fallback: " + dbg_last)

                if not voice_lines_list:
                    voice_lines_list = pick_unique_lines_from_pool(csv_pool, need, batch_forbid_exact, batch_forbid_bow)

                disclaimer_line = pick_unique_disclaimer(disclaimers_p2, batch_forbid_exact, batch_forbid_bow)
                extra_line = pick_unique_extra_point(batch_forbid_exact, batch_forbid_bow)
                voice_lines_list = voice_lines_list + [disclaimer_line, extra_line]

            voice_lines = "\n".join([sanitize_for_sora(x) for x in voice_lines_list if compact_spaces(x)])

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
                st.text_area("Prompt", prompts[i], height=650, key=f"view_{i}")
                copy_button_unicode_safe(prompts[i], key=f"copy_{i}")
else:
    st.warning("Upload a shoe image to begin.")

st.divider()
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.used_voice_sig_exact.clear()
    st.session_state.used_voice_sig_bow.clear()
    st.session_state.generated_prompts = []
    st.success("Reset done.")
