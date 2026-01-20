import streamlit as st
import pandas as pd
import random
import base64
import re
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 - Total 10s - Multi-scene - Anti-duplicate - TikTok Shop SAFE - SORA ASCII SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"

SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]
REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

# =========================
# COPY BUTTON (1 CLICK)
# =========================
def copy_button(text: str, key: str):
    b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    html = f"""
    <button id="{key}" style="
        padding:8px 14px;border-radius:10px;border:1px solid #ccc;
        cursor:pointer;background:#fff;font-weight:700;">COPY</button>
    <span id="{key}_s" style="margin-left:8px;font-size:12px;"></span>
    <script>
    const btn = document.getElementById("{key}");
    const s = document.getElementById("{key}_s");
    btn.onclick = async () => {{
        try {{
            await navigator.clipboard.writeText(atob("{b64}"));
            s.innerText = "Copied";
            setTimeout(()=>s.innerText="",1500);
        }} catch(e) {{
            s.innerText = "Clipboard blocked";
            setTimeout(()=>s.innerText="",2500);
        }}
    }}
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
# LOAD CSV
# =========================
@st.cache_data
def load_dialogues():
    df = pd.read_csv("dialogue_library.csv")
    df.columns = [c.strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv")
    df.columns = [c.strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_disclaimer_prompt2_flexible():
    df = pd.read_csv("disclaimer_prompt2.csv")
    cols = [c.strip() for c in df.columns.tolist()]
    df.columns = cols

    if "disclaimer" in cols:
        arr = df["disclaimer"].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    preferred = ["text", "mien_tru", "mien tru", "note", "content", "noi_dung", "line", "script"]
    for c in preferred:
        if c in cols:
            arr = df[c].dropna().astype(str).tolist()
            return [x.strip() for x in arr if x.strip()]

    if len(cols) >= 2 and cols[0].lower() in ["id", "stt", "no"]:
        arr = df[cols[1]].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    last = cols[-1]
    arr = df[last].dropna().astype(str).tolist()
    return [x.strip() for x in arr if x.strip()]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2_flexible()

# =========================
# SESSION - ANTI DUP
# =========================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

# =========================
# UTILS
# =========================
def safe_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s

def to_ascii(s: str) -> str:
    """
    Convert Vietnamese/Unicode text to ASCII-safe text (remove accents).
    """
    if not s:
        return ""
    s = s.replace("Đ", "D").replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # keep basic ascii, replace odd spaces
    s = s.replace("\u00a0", " ")
    return s

def clean_ascii_line(s: str) -> str:
    s = to_ascii(s)
    # remove non-ascii leftovers
    s = s.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pick_unique(pool, used_ids: set, key: str):
    def get_id(x):
        val = safe_text(x.get(key))
        if val:
            return val
        return str(hash(str(x)))

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
    out = []
    for _ in range(n):
        s = pick_unique(pool, st.session_state.used_scene_ids, "id")
        out.append(s)
    return out

# =========================
# VN -> EN ASCII mapping (for Sora stability)
# =========================
LIGHTING_MAP = {
    "anh vang xien": "warm side light",
    "anh vang": "warm light",
    "anh trang": "clean white light",
    "anh tu nhien": "natural daylight",
}
LOCATION_MAP = {
    "duong pho sang som": "morning street",
    "san tap": "training ground",
    "quan ca phe": "daylight cafe",
    "studio": "clean studio",
}
WEATHER_MAP = {
    "suong nhe": "light mist",
    "gio nhe": "light breeze",
    "nang diu": "soft sunlight",
}
MOOD_MAP = {
    "nang dong": "energetic",
    "manh me": "strong",
    "lang man": "calm romantic",
}
MOTION_MAP = {
    "orbit cham": "slow orbit",
    "tracking nhanh": "fast tracking",
    "crane xuong": "crane down",
    "pan ngang": "horizontal pan",
    "wide dolly": "wide dolly",
}

def map_field(v: str, mapping: dict, fallback_prefix: str = "") -> str:
    vv = clean_ascii_line(v).lower()
    if vv in mapping:
        return mapping[vv]
    # try fuzzy contains
    for k, out in mapping.items():
        if k and k in vv:
            return out
    if not vv:
        return ""
    return (fallback_prefix + vv).strip()

def scene_to_en_ascii(scene: dict) -> dict:
    lighting = map_field(safe_text(scene.get("lighting")), LIGHTING_MAP)
    location = map_field(safe_text(scene.get("location")), LOCATION_MAP)
    weather = map_field(safe_text(scene.get("weather")), WEATHER_MAP)
    mood = map_field(safe_text(scene.get("mood")), MOOD_MAP)
    motion = map_field(safe_text(scene.get("motion")), MOTION_MAP)

    # fallback: use ascii cleaned vn text if mapping misses
    if not lighting:
        lighting = clean_ascii_line(safe_text(scene.get("lighting")))
    if not location:
        location = clean_ascii_line(safe_text(scene.get("location")))
    if not weather:
        weather = clean_ascii_line(safe_text(scene.get("weather")))
    if not mood:
        mood = clean_ascii_line(safe_text(scene.get("mood")))
    if not motion:
        motion = clean_ascii_line(safe_text(scene.get("motion")))

    return {
        "lighting": lighting,
        "location": location,
        "weather": weather,
        "mood": mood,
        "motion": motion,
    }

# =========================
# FILENAME HEURISTIC (fallback only)
# =========================
def detect_shoe_from_filename(name: str) -> str:
    n = (name or "").lower()
    rules = [
        ("boots",   ["boot", "chelsea", "combat", "martin"]),
        ("sandals", ["sandal", "sandals", "dep", "dep", "slipper", "slides"]),
        ("leather", ["loafer", "loafers", "moc", "moccasin", "horsebit", "oxford", "derby", "tassel", "brogue",
                     "giaytay", "giay tay", "giay_da", "giayda", "dressshoe"]),
        ("runner",  ["runner", "running", "jog", "marathon", "gym", "train", "thethao", "the thao", "sport"]),
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

# =========================
# GEMINI VISION DETECT (AI priority)
# =========================
def gemini_detect_shoe_type(img: Image.Image, api_key: str) -> Tuple[Optional[str], str]:
    """
    Returns (shoe_type or None, raw_text).
    Uses list_models to avoid 404 NotFound and handles quota/rate issues by fallback.
    """
    api_key = (api_key or "").strip()
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
            "models/gemini-2.5-flash",
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
            "You are a shoe classification system.\n"
            "Return ONLY ONE label from this list:\n"
            f"{', '.join(SHOE_TYPES)}\n\n"
            "Rules:\n"
            "- Return exactly one word.\n"
            "- No explanations.\n"
            "- If dress shoe / loafer / oxford / derby => leather.\n"
            "- If sports sneaker => sneaker or runner.\n"
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
        msg = f"{type(e).__name__}: {e}"
        # common: ResourceExhausted / quota
        return None, f"CALL_FAIL: {msg}"

# =========================
# DIALOGUE BANK (VN stored, output will be ASCII)
# =========================
TONE_BANK = {
    "Tu tin": {
        "open": [
            "Hom nay minh chon kieu gon gang de ra ngoai cho tu tin hon.",
            "Minh thich cam giac buoc di gon gon va co nhip.",
            "Minh uu tien tong the sach, de phoi va nhin sang dang."
        ],
        "mid": [
            "Di mot luc thay nhip buoc deu, cam giac kha on dinh.",
            "Minh thay form len chan nhin gon, de di suot ngay.",
            "Cam giac di chuyen nhe nhang, khong bi roi mat."
        ],
        "close": [
            "Tong the don gian nhung co diem tinh te rieng.",
            "Minh thich kieu toi gian de tao phong cach.",
            "Voi minh, gon gang la du dep roi."
        ],
    },
    "Truyen cam": {
        "open": [
            "Co nhung doi mang vao la thay tam trang diu lai lien.",
            "Minh thich cam giac nhe nhang, cham rai ma van chi chu.",
            "Nhin ky moi thay cai hay nam o su tinh gian."
        ],
        "mid": [
            "Di cham thoi nhung cam giac lai rat thu tha.",
            "Minh thich nhip buoc em, tao cam giac de chiu.",
            "Cang nhin cang thay tong the hai hoa."
        ],
        "close": [
            "Moi buoc nhu giu lai mot chut binh yen.",
            "Vua du tinh te de nhin lau khong chan.",
            "Doi khi chi can vay la dep."
        ],
    },
    "Manh me": {
        "open": [
            "Hom nay minh muon nhip buoc dut khoat hon mot chut.",
            "Minh thich cam giac chac chan khi di chuyen nhanh.",
            "Ngay ban ron thi minh can su gon va on dinh."
        ],
        "mid": [
            "Di nhanh van thay kiem soat tot, khong bi chong chenh.",
            "Nhip buoc chac, cam giac bam chan on.",
            "Cam giac gon gang giup minh tu tin hon khi di chuyen."
        ],
        "close": [
            "Tong the nhin khoe ma van sach.",
            "Gon, chac, de phoi, dung gu minh.",
            "Chi can on dinh la minh yen tam."
        ],
    },
    "Lang man": {
        "open": [
            "Chieu nay ra ngoai chut, tu nhien mood nhe hon.",
            "Minh thich kieu di cham, nhin moi thu mem lai.",
            "Nhung ngay nhu vay, minh uu tien cam giac thu tha."
        ],
        "mid": [
            "Nhip buoc nhe, nhin tong the rat hai hoa.",
            "Cam giac vua van khien minh muon di them mot doan nua.",
            "Don gian thoi nhung len hinh lai thay rat diu."
        ],
        "close": [
            "Cang toi gian cang de tao cam xuc rieng.",
            "Minh thich su tinh te nam o nhung thu gian don.",
            "Mot chut nhe nhang la du."
        ],
    },
    "Tu nhien": {
        "open": [
            "Minh uu tien thoai mai, kieu mang la muon di tiep.",
            "Hom nay minh chon phong cach tu nhien, khong cau ky.",
            "Di ra ngoai ma van thay nhe nhang la minh thich."
        ],
        "mid": [
            "Cam giac di chuyen mem, de chiu.",
            "Nhin tong the rat tu nhien, khong bi gong.",
            "Minh thay hop nhung ngay muon tha long."
        ],
        "close": [
            "Gon gang vay thoi nhung lai de dung hang ngay.",
            "Minh thich kieu don gian ma nhin sach.",
            "Nhe nhang la du dep roi."
        ],
    }
}

TONE_UI_MAP = {
    "Truyen cam": "Truyen cam",
    "Tu tin": "Tu tin",
    "Manh me": "Manh me",
    "Lang man": "Lang man",
    "Tu nhien": "Tu nhien",
}

def split_sentences(text: str) -> List[str]:
    t = safe_text(text)
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"[.!?]+", t) if p.strip()]
    return parts

def get_dialogue_from_csv(row: dict) -> str:
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung", "noi dung"]:
        if col in row and safe_text(row.get(col)):
            return safe_text(row.get(col))
    return ""

def ensure_period(x: str) -> str:
    x = clean_ascii_line(x)
    if not x:
        return ""
    if x.endswith((".", "!", "?")):
        return x
    return x + "."

def get_dialogue_3_sentences(row: dict, tone_key: str) -> str:
    """
    Prompt 1: exactly 3 ASCII lines.
    """
    bank = TONE_BANK.get(tone_key, TONE_BANK["Tu tin"])
    candidate = get_dialogue_from_csv(row)
    if candidate:
        candidate = clean_ascii_line(candidate)
        parts = split_sentences(candidate)
        parts = [clean_ascii_line(p) for p in parts if clean_ascii_line(p)]
    else:
        parts = []

    if len(parts) >= 3:
        a, b, c = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        a, b = parts[0], parts[1]
        c = random.choice(bank["close"])
    elif len(parts) == 1:
        a = parts[0]
        b = random.choice(bank["mid"])
        c = random.choice(bank["close"])
    else:
        a = random.choice(bank["open"])
        b = random.choice(bank["mid"])
        c = random.choice(bank["close"])

    if b.lower() == a.lower():
        b = random.choice(bank["mid"])
    if c.lower() in {a.lower(), b.lower()}:
        c = random.choice(bank["close"])

    a, b, c = ensure_period(a), ensure_period(b), ensure_period(c)
    return f"{a}\n{b}\n{c}"

def get_dialogue_2_sentences(row: dict, tone_key: str) -> str:
    """
    Prompt 2: exactly 2 ASCII lines. Disclaimer will be line 3.
    """
    bank = TONE_BANK.get(tone_key, TONE_BANK["Tu tin"])
    candidate = get_dialogue_from_csv(row)
    if candidate:
        candidate = clean_ascii_line(candidate)
        parts = split_sentences(candidate)
        parts = [clean_ascii_line(p) for p in parts if clean_ascii_line(p)]
    else:
        parts = []

    if len(parts) >= 2:
        a, b = parts[0], parts[1]
    elif len(parts) == 1:
        a = parts[0]
        b = random.choice(bank["mid"])
    else:
        a = random.choice(bank["open"])
        b = random.choice(bank["mid"])

    if b.lower() == a.lower():
        b = random.choice(bank["mid"])

    a, b = ensure_period(a), ensure_period(b)
    return f"{a}\n{b}"

def short_disclaimer_ascii(raw: str) -> str:
    s = clean_ascii_line(raw)
    if not s:
        s = "Noi dung chi mang tinh chia se trai nghiem."
    s = ensure_period(s)
    # keep it short
    if len(s) > 140:
        s = s[:140].rstrip() + "."
    return s

# =========================
# PROMPT BUILDER (ASCII SAFE)
# =========================
def build_prompt_unified(
    mode: str,  # "p1" or "p2"
    shoe_type: str,
    shoe_name: str,
    scene_list: List[dict],
    timeline: List[Tuple[float, float]],
    voice_lines: str,
) -> str:
    shoe_name_a = clean_ascii_line(shoe_name)
    shoe_type_a = clean_ascii_line(shoe_type)

    if mode == "p1":
        title = "PROMPT 1 - NO CAMEO"
        cast_rule = (
            "CAST RULE\n"
            "No people on screen\n"
            "No cameo\n"
            f"Voice ID: {CAMEO_VOICE_ID}\n"
        )
    else:
        title = "PROMPT 2 - WITH CAMEO"
        cast_rule = (
            "CAST RULE\n"
            "Cameo appears naturally like a phone video review\n"
            f"Cameo and Voice ID: {CAMEO_VOICE_ID}\n"
            "No hard call to action, no price, no discount, no guarantees\n"
        )

    # scene lines
    scene_lines = []
    for i, (sc, (a, b)) in enumerate(zip(scene_list, timeline), start=1):
        en = scene_to_en_ascii(sc)
        scene_lines.append(
            f"{a:.1f}-{b:.1f}s: location {en['location']}, lighting {en['lighting']}, motion {en['motion']}, "
            f"weather {en['weather']}, mood {en['mood']}"
        )

    scene_block = "\n".join(scene_lines)

    prompt = f"""SORA VIDEO PROMPT - {title} - TOTAL 10s

VIDEO SETUP
Vertical 9:16
Total duration exactly 10 seconds
Ultra sharp 4K output
Realistic motion, not a still image
No on screen text
No logo
No watermark
No blur, haze, or glow

{cast_rule}
SHOE REFERENCE LOCK
Use only the uploaded shoe image as reference
Keep 100 percent shoe identity: toe shape, panels, stitching, sole, proportions
No redesign, no deformation, no guessing, no color shift
Lace rule: if the reference shoe has laces then keep laces in all frames; if no laces then absolutely no laces

PRODUCT
shoe_name: {shoe_name_a}
shoe_type: {shoe_type_a}

SCENES INSIDE 10s
{scene_block}

AUDIO TIMELINE
0.0-1.2s: no voice, light ambient only
1.2-6.9s: voice on
6.9-10.0s: voice off, gentle fade out

VOICEOVER 1.2-6.9s
{voice_lines}
"""
    # ensure pure ascii output
    prompt = clean_ascii_line(prompt.replace("\r\n", "\n").replace("\r", "\n"))
    # but we must keep newlines; clean_ascii_line collapses them, so do a safer ascii pass:
    prompt = to_ascii(prompt)
    prompt = prompt.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    # restore readable newlines by rebuilding from original template approach:
    # Instead of collapsing, do per-line cleaning:
    lines = []
    for line in f"""SORA VIDEO PROMPT - {title} - TOTAL 10s

VIDEO SETUP
Vertical 9:16
Total duration exactly 10 seconds
Ultra sharp 4K output
Realistic motion, not a still image
No on screen text
No logo
No watermark
No blur, haze, or glow

{cast_rule}
SHOE REFERENCE LOCK
Use only the uploaded shoe image as reference
Keep 100 percent shoe identity: toe shape, panels, stitching, sole, proportions
No redesign, no deformation, no guessing, no color shift
Lace rule: if the reference shoe has laces then keep laces in all frames; if no laces then absolutely no laces

PRODUCT
shoe_name: {shoe_name_a}
shoe_type: {shoe_type_a}

SCENES INSIDE 10s
{scene_block}

AUDIO TIMELINE
0.0-1.2s: no voice, light ambient only
1.2-6.9s: voice on
6.9-10.0s: voice off, gentle fade out

VOICEOVER 1.2-6.9s
{voice_lines}
""".splitlines():
        lines.append(clean_ascii_line(line))
    return "\n".join(lines).strip()

# =========================
# SIDEBAR: GEMINI KEY (session only)
# =========================
with st.sidebar:
    st.markdown("### Gemini API Key (optional)")
    st.caption("Used for AI shoe_type detection. If AI fails, filename fallback is used.")

    api_key_input = st.text_input("GEMINI_API_KEY", value=st.session_state.gemini_api_key, type="password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save key (session)", use_container_width=True):
            st.session_state.gemini_api_key = api_key_input.strip()
            st.success("Saved in current session.")
    with c2:
        if st.button("Clear key", use_container_width=True):
            st.session_state.gemini_api_key = ""
            st.info("Cleared.")

    if st.session_state.gemini_api_key:
        st.success("Key is set (session).")
    else:
        st.warning("No key set (AI will not run).")

# =========================
# UI
# =========================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
    mode = st.radio("Prompt mode", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone_ui = st.selectbox("Tone", ["Truyen cam", "Tu tin", "Manh me", "Lang man", "Tu nhien"], index=1)
    scene_count = st.slider("Scene count (inside total 10s)", 2, 4, 3)
    count = st.slider("Prompts per click", 1, 10, 5)

with right:
    st.subheader("Quick guide")
    st.write("1) Upload image 2) Choose Prompt 1/2 3) Choose tone 4) Choose scene count 5) Generate 6) Copy")
    st.caption("Dialogue columns: " + ", ".join(dialogue_cols))
    st.caption("Scene columns: " + ", ".join(scene_cols))
    st.info("Total video duration is ALWAYS 10 seconds. Scenes are split inside 10s.")
    st.info("Prompt 1: 3 voice lines (no disclaimer). Prompt 2: 2 voice lines + 1 short disclaimer (3 total lines).")
    st.info("Output is ASCII-only to avoid Sora paste text corruption.")

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

    detect_mode = st.selectbox(
        "shoe_type mode",
        ["AI (image) - preferred", "Auto (filename) - fallback", "Manual"],
        index=0
    )

    detected_filename = detect_shoe_from_filename(uploaded.name)

    detected_ai = None
    raw_ai = ""
    if detect_mode.startswith("AI"):
        detected_ai, raw_ai = gemini_detect_shoe_type(img, st.session_state.gemini_api_key)

    if detect_mode.startswith("AI"):
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

    btn_label = "Generate PROMPT 1" if mode.startswith("PROMPT 1") else "Generate PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        tone_key = TONE_UI_MAP.get(tone_ui, "Tu tin")

        for _ in range(count):
            d_pool = filter_dialogues(shoe_type, tone_ui)
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)

            if mode.startswith("PROMPT 1"):
                voice_lines = get_dialogue_3_sentences(d, tone_key)
                p = build_prompt_unified("p1", shoe_type, shoe_name, scene_list, timeline, voice_lines)
            else:
                voice_2 = get_dialogue_2_sentences(d, tone_key)
                disclaimer_raw = random.choice(disclaimers_p2) if disclaimers_p2 else "Noi dung chi mang tinh chia se trai nghiem."
                disclaimer = short_disclaimer_ascii(disclaimer_raw)
                voice_lines = f"{voice_2}\n{disclaimer}"
                p = build_prompt_unified("p2", shoe_type, shoe_name, scene_list, timeline, voice_lines)

            arr.append(p)

        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.markdown("Output prompts")
        tabs = st.tabs([str(i + 1) for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=420, key=f"view_{i}")
                copy_button(prompts[i], key=f"copy_view_{i}")

else:
    st.warning("Upload a shoe image to begin.")

st.divider()
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.generated_prompts = []
    st.success("Reset done.")
