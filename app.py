import streamlit as st
import pandas as pd
import random
import re
import json
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

# ============================================================
# Sora Prompt Studio Pro - Director Edition
# FIX: remove JS clipboard copy (iframe clipboard can corrupt)
# Use native copy in text_area + download .txt UTF-8
# Persistent Gemini key in browser localStorage via query param bridge
# ============================================================

st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 | Total 10s | Multi-shot | Anti-duplicate | TikTok Shop SAFE | Copy Safe Unicode")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]
REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

# =========================
# TEXT NORMALIZE (HARD CLEAN)
# =========================
ZERO_WIDTH_PATTERN = r"[\u200b\u200c\u200d\uFEFF]"
CONTROL_PATTERN = r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]"
NBSP = "\u00A0"
LSEP = "\u2028"
PSEP = "\u2029"

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)

    try:
        s = unicodedata.normalize("NFC", s)
    except Exception:
        pass

    # remove zero width chars / BOM
    s = re.sub(ZERO_WIDTH_PATTERN, "", s)

    # normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # replace special separators
    s = s.replace(LSEP, "\n").replace(PSEP, "\n")

    # replace NBSP
    s = s.replace(NBSP, " ")

    # remove control chars except \n and \t
    # first protect \n and \t
    s = s.replace("\n", "__NL__").replace("\t", "__TB__")
    s = re.sub(CONTROL_PATTERN, "", s)
    s = s.replace("__NL__", "\n").replace("__TB__", "\t")

    return s

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
    s = compact_spaces(str(v))
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

# =========================
# localStorage bridge for key (no server save)
# Streamlit cannot read localStorage directly, so we:
# - on load: inject key into URL param GEMINI_API_KEY
# - python reads st.query_params
# =========================
def localstorage_key_bridge(key_name="GEMINI_API_KEY"):
    import streamlit.components.v1 as components
    html = f"""
    <script>
    (function() {{
        const KEY = "{key_name}";
        const url = new URL(window.location.href);

        if (!url.searchParams.get(KEY)) {{
            const saved = localStorage.getItem(KEY);
            if (saved) {{
                url.searchParams.set(KEY, saved);
                window.history.replaceState(null, "", url.toString());
            }}
        }}

        window.__saveKey = function(val) {{
            try {{
                if (val) {{
                    localStorage.setItem(KEY, val);
                    return "OK";
                }}
                return "EMPTY";
            }} catch(e) {{
                return "FAIL";
            }}
        }}

        window.__clearKey = function() {{
            try {{
                localStorage.removeItem(KEY);
                return "OK";
            }} catch(e) {{
                return "FAIL";
            }}
        }}
    }})();
    </script>
    """
    components.html(html, height=0)

def js_save_key(value: str):
    import streamlit.components.v1 as components
    payload = json.dumps(normalize_text(value))
    components.html(f"<script>window.__saveKey({payload});</script>", height=0)

def js_clear_key():
    import streamlit.components.v1 as components
    components.html("<script>window.__clearKey();</script>", height=0)

# =========================
# FILE CHECK
# =========================
missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing) + " (must be in same folder as app.py)")
    st.stop()

# =========================
# LOAD CSV (UTF-8 BOM SAFE)
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
def load_disclaimer_prompt2_flexible():
    df = read_csv_flexible("disclaimer_prompt2.csv")
    cols = [c.strip() for c in df.columns.tolist()]
    df.columns = cols

    if "disclaimer" in cols:
        arr = df["disclaimer"].dropna().astype(str).tolist()
        return [compact_spaces(x) for x in arr if compact_spaces(x)]

    preferred = ["text", "mien_tru", "miễn_trừ", "note", "content", "noi_dung", "line", "script"]
    for c in preferred:
        if c in cols:
            arr = df[c].dropna().astype(str).tolist()
            return [compact_spaces(x) for x in arr if compact_spaces(x)]

    if len(cols) >= 2 and cols[0].lower() in ["id", "stt", "no"]:
        arr = df[cols[1]].dropna().astype(str).tolist()
        return [compact_spaces(x) for x in arr if compact_spaces(x)]

    last = cols[-1]
    arr = df[last].dropna().astype(str).tolist()
    return [compact_spaces(x) for x in arr if compact_spaces(x)]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2_flexible()

# =========================
# SESSION
# =========================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

# auto-load from query param (localStorage bridge writes it to URL)
params = st.query_params
if "GEMINI_API_KEY" in params and not st.session_state.gemini_api_key:
    st.session_state.gemini_api_key = normalize_text(params["GEMINI_API_KEY"]).strip()

# =========================
# UTILS
# =========================
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
        out.append(pick_unique(pool, st.session_state.used_scene_ids, "id"))
    return out

# =========================
# FILENAME HEURISTIC
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

# =========================
# GEMINI VISION DETECT (AUTO PICK MODEL)
# =========================
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

        models = genai.list_models()
        available = []
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append(m.name)

        if not available:
            return None, "NO_MODELS"

        preferred = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro-vision"]
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
            f"{', '.join(SHOE_TYPES)}\n"
            "Rules:\n"
            "- Return exactly one word.\n"
            "- No explanations.\n"
            "- Dress shoe, loafer, oxford, derby => leather.\n"
            "- Sports sneaker => sneaker or runner.\n"
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
# DIALOGUE BANK
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
    },
    "Truyền cảm": {
        "open": [
            "Có những đôi mang vào là thấy tâm trạng dịu lại liền.",
            "Mình thích cảm giác nhẹ nhàng, chậm rãi mà vẫn chỉn chu.",
            "Nhìn kỹ mới thấy cái hay nằm ở sự tinh giản."
        ],
        "mid": [
            "Đi chậm thôi nhưng cảm giác lại rất thư thả.",
            "Mình thích nhịp bước êm, tạo cảm giác dễ chịu.",
            "Càng nhìn càng thấy tổng thể hài hòa."
        ],
        "close": [
            "Mỗi bước như giữ lại một chút bình yên.",
            "Vừa đủ tinh tế để nhìn lâu không chán.",
            "Đôi khi chỉ cần vậy là đẹp."
        ],
    },
    "Mạnh mẽ": {
        "open": [
            "Hôm nay mình muốn nhịp bước dứt khoát hơn một chút.",
            "Mình thích cảm giác chắc chân khi di chuyển nhanh.",
            "Ngày bận rộn thì mình cần sự gọn và ổn định."
        ],
        "mid": [
            "Đi nhanh vẫn thấy kiểm soát tốt, không bị chông chênh.",
            "Nhịp bước chắc, cảm giác bám chân ổn.",
            "Cảm giác gọn gàng giúp mình tự tin hơn khi di chuyển."
        ],
        "close": [
            "Tổng thể nhìn khỏe mà vẫn sạch.",
            "Gọn gàng, chắc chắn, dễ phối, đúng gu mình.",
            "Chỉ cần ổn định là mình yên tâm."
        ],
    },
    "Lãng mạn": {
        "open": [
            "Chiều nay ra ngoài chút, tự nhiên mood nhẹ hơn.",
            "Mình thích kiểu đi chậm, nhìn mọi thứ mềm lại.",
            "Những ngày như vậy, mình ưu tiên cảm giác thư thả."
        ],
        "mid": [
            "Nhịp bước nhẹ, nhìn tổng thể rất hài hòa.",
            "Cảm giác vừa vặn khiến mình muốn đi thêm một đoạn nữa.",
            "Đơn giản thôi nhưng lên hình lại thấy rất dịu."
        ],
        "close": [
            "Càng tối giản càng dễ tạo cảm xúc riêng.",
            "Mình thích sự tinh tế nằm ở những thứ giản đơn.",
            "Một chút nhẹ nhàng là đủ."
        ],
    },
    "Tự nhiên": {
        "open": [
            "Mình ưu tiên thoải mái, kiểu mang là muốn đi tiếp.",
            "Hôm nay mình chọn phong cách tự nhiên, không cầu kỳ.",
            "Đi ra ngoài mà vẫn thấy nhẹ nhàng là mình thích."
        ],
        "mid": [
            "Cảm giác di chuyển mềm, dễ chịu.",
            "Nhìn tổng thể rất tự nhiên, không bị gồng.",
            "Mình thấy hợp những ngày muốn thả lỏng."
        ],
        "close": [
            "Gọn gàng vậy thôi nhưng lại dễ dùng hằng ngày.",
            "Mình thích kiểu đơn giản mà nhìn sạch.",
            "Nhẹ nhàng là đủ đẹp rồi."
        ],
    }
}

def split_sentences(text: str) -> List[str]:
    t = safe_text(text)
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"[.!?]+", t) if p.strip()]
    return parts

def get_dialogue_from_csv(row: dict) -> str:
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row and safe_text(row.get(col)):
            return safe_text(row.get(col))
    return ""

def get_dialogue_3_sentences(row: dict, tone: str) -> str:
    candidate = get_dialogue_from_csv(row)
    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])

    if candidate:
        parts = split_sentences(candidate)
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
    else:
        a = random.choice(bank["open"])
        b = random.choice(bank["mid"])
        c = random.choice(bank["close"])

    if b.strip().lower() == a.strip().lower():
        b = random.choice(bank["mid"])
    if c.strip().lower() in {a.strip().lower(), b.strip().lower()}:
        c = random.choice(bank["close"])

    a, b, c = ensure_end_punct(a), ensure_end_punct(b), ensure_end_punct(c)
    return normalize_text(f"{a}\n{b}\n{c}")

def get_dialogue_2_sentences(row: dict, tone: str) -> str:
    candidate = get_dialogue_from_csv(row)
    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])

    if candidate:
        parts = split_sentences(candidate)
        if len(parts) >= 2:
            a, b = parts[0], parts[1]
        elif len(parts) == 1:
            a = parts[0]
            b = random.choice(bank["mid"])
        else:
            a = random.choice(bank["open"])
            b = random.choice(bank["mid"])
    else:
        a = random.choice(bank["open"])
        b = random.choice(bank["mid"])

    if b.strip().lower() == a.strip().lower():
        b = random.choice(bank["mid"])

    a, b = ensure_end_punct(a), ensure_end_punct(b)
    return normalize_text(f"{a}\n{b}")

def short_disclaimer(raw: str) -> str:
    s = compact_spaces(raw)
    if not s:
        s = "Nội dung chỉ mang tính chia sẻ trải nghiệm."
    s = ensure_end_punct(s)
    if len(s) > 180:
        s = s[:180].rstrip() + "."
    return normalize_text(s)

# =========================
# PROMPT BUILDER (STYLE CLOSE TO YOUR OLD PROMPT, ASCII SEPARATORS ONLY)
# NOTE: no box drawing, no emojis
# =========================
def build_prompt_unified(
    mode: str,            # "p1" or "p2"
    shoe_type: str,
    shoe_name: str,
    scene_list: List[dict],
    timeline: List[Tuple[float, float]],
    voice_lines: str,
) -> str:
    shoe_name = compact_spaces(shoe_name)
    shoe_type = compact_spaces(shoe_type)
    voice_lines = normalize_text(voice_lines)

    if mode == "p1":
        header = "SORA VIDEO PROMPT - PROMPT 1 (NO CAMEO)"
        cast = (
            "CAST RULE\n"
            "No people on screen\n"
            "No cameo\n"
            f"Voice ID: {CAMEO_VOICE_ID}\n"
        )
    else:
        header = "SORA VIDEO PROMPT - PROMPT 2 (WITH CAMEO)"
        cast = (
            "CAST RULE\n"
            "Cameo appears naturally like a phone video review\n"
            f"Cameo and Voice ID: {CAMEO_VOICE_ID}\n"
            "No price, no discount, no guarantees\n"
            "No hard call to action\n"
        )

    shot_lines = []
    for sc, (a, b) in zip(scene_list, timeline):
        loc = compact_spaces(safe_text(sc.get("location")))
        light = compact_spaces(safe_text(sc.get("lighting")))
        mot = compact_spaces(safe_text(sc.get("motion")))
        wea = compact_spaces(safe_text(sc.get("weather")))
        mood = compact_spaces(safe_text(sc.get("mood")))
        shot_lines.append(
            f"{a:.1f}-{b:.1f}s: location {loc}. lighting {light}. motion {mot}. weather {wea}. mood {mood}."
        )
    shots = "\n".join(shot_lines)

    prompt = f"""
{CAMEO_VOICE_ID}

{header}

VIDEO SETUP
Vertical 9:16
Total duration exactly 10 seconds
Ultra sharp 4K output
Realistic cinematic video (not a still image)
Use the exact uploaded shoe image
No on screen text
No logo
No watermark
No blur
No haze
No glow

{cast}

SHOE REFERENCE LOCK (CRITICAL)
Use only the uploaded shoe image as reference
Lock 100 percent shoe identity: toe shape, vamp, panels, stitching, sole, proportions
No redesign, no deformation, no guessing, no color shift
Lace rule:
If the shoe has laces then keep laces in all frames
If the shoe has no laces then absolutely no laces

PRODUCT
shoe_name: {shoe_name}
shoe_type: {shoe_type}

TIMELINE
0.0-1.2s: no voice, show shoe clearly
1.2-6.9s: voice on
6.9-10.0s: voice off, gentle fade out

SHOTS (INSIDE 10s)
{shots}

VOICEOVER (1.2-6.9s)
{voice_lines}
"""
    return normalize_text(prompt).strip()

# =========================
# SIDEBAR: GEMINI KEY (PERSIST IN BROWSER)
# =========================
with st.sidebar:
    st.markdown("### Gemini API Key (optional)")
    st.caption("Saved in your browser localStorage (no server save).")

    localstorage_key_bridge("GEMINI_API_KEY")

    api_key_input = st.text_input("GEMINI_API_KEY", value=st.session_state.gemini_api_key, type="password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save key (remember)", use_container_width=True):
            st.session_state.gemini_api_key = api_key_input.strip()
            js_save_key(st.session_state.gemini_api_key)
            st.success("Saved in this browser.")
    with c2:
        if st.button("Clear saved key", use_container_width=True):
            st.session_state.gemini_api_key = ""
            js_clear_key()
            st.info("Cleared from this browser.")

    if st.session_state.gemini_api_key:
        st.success("Key is set.")
    else:
        st.warning("No key set. AI detect will not run.")

# =========================
# UI
# =========================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
    mode = st.radio("Prompt mode", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone = st.selectbox("Tone", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    scene_count = st.slider("Shots inside total 10s", 2, 4, 4)
    count = st.slider("Prompts per click", 1, 10, 5)

with right:
    st.subheader("How to paste into Sora (no lỗi)")
    st.write("Cách 1 (khuyên dùng): Click vào ô Prompt, nhấn Ctrl+A rồi Ctrl+C. Dán vào Sora.")
    st.write("Cách 2: Bấm Download .txt, mở file rồi copy sang Sora.")
    st.caption("Dialogue columns: " + ", ".join([str(x) for x in dialogue_cols]))
    st.caption("Scene columns: " + ", ".join([str(x) for x in scene_cols]))
    st.info("Prompt 1: 3 dòng thoại. Prompt 2: 2 dòng thoại + 1 dòng miễn trừ.")

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
            st.caption("AI raw: " + normalize_text(raw_ai))
        else:
            shoe_type = detected_filename
            st.warning("AI shoe_type failed. Using filename fallback.")
            st.caption("AI raw: " + normalize_text(raw_ai))
            st.info("Fallback shoe_type: " + detected_filename)
    elif detect_mode.startswith("Auto"):
        shoe_type = detected_filename
        st.info("Filename shoe_type: " + shoe_type)
    else:
        shoe_type = st.selectbox(
            "Manual shoe_type",
            SHOE_TYPES,
            index=SHOE_TYPES.index("leather") if "leather" in SHOE_TYPES else 0
        )
        st.success("Manual shoe_type: " + shoe_type)

    st.caption("shoe_name: " + normalize_text(shoe_name))

    btn_label = "Generate PROMPT 1" if mode.startswith("PROMPT 1") else "Generate PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            d_pool = filter_dialogues(shoe_type, tone)
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
        st.markdown("Output prompts")
        tabs = st.tabs([str(i + 1) for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.write("Copy theo cách an toàn: click ô dưới -> Ctrl+A -> Ctrl+C (không dùng nút JS).")
                st.text_area("Prompt", prompts[i], height=560, key=f"view_{i}")

                st.download_button(
                    label="Download prompt .txt (UTF-8)",
                    data=prompts[i].encode("utf-8"),
                    file_name=f"prompt_{i+1}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
else:
    st.warning("Upload a shoe image to begin.")

st.divider()

if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.generated_prompts = []
    st.success("Reset done.")
