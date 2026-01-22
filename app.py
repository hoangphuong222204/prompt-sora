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
# Cloud-safe: NO file writing, NO hard-coded API keys
# - Total 10s prompt
# - 2 modes: Prompt 1 (no cameo) / Prompt 2 (with cameo)
# - 2–4 shots inside 10s (default 4)
# - 5 prompts per click (default)
# - 5 different visual styles per batch (style pack)
# - Voice style pack (đa dạng cách đọc)
# - Unicode copy-safe (không lỗi dấu)
# - Text/logo orientation lock (no mirrored/reversed logo)
# - Gemini API key nhập trực tiếp trên UI (session only)
# - CSV flexible: dùng file trong repo nếu có; nếu không có thì cho upload CSV trong app;
#   nếu vẫn không có thì dùng "built-in mini library" để app luôn chạy được.
# ==========================================================

st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 • Total 10s • Multi-shot • Anti-duplicate • TikTok Shop SAFE • Copy Safe Unicode • Style Pack • Key-in-UI")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

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
    payload = json.dumps(text, ensure_ascii=False)
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
# FALLBACK MINI LIBRARIES (so app always runs)
# =========================
BUILTIN_DIALOGUES = [
    {"id":"vn_001","tone":"Tự tin","shoe_type":"leather","dialogue":"Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn. Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định. Nhìn tổng thể tối giản nhưng vẫn có điểm tinh tế."},
    {"id":"vn_002","tone":"Tự nhiên","shoe_type":"sneaker","dialogue":"Mình thích kiểu nhìn sạch và dễ phối. Bước đi thấy khá nhẹ, nhịp chân thoải mái. Tổng thể gọn, nhìn lên form ổn."},
    {"id":"vn_003","tone":"Mạnh mẽ","shoe_type":"runner","dialogue":"Mình ưu tiên cảm giác chắc chân khi di chuyển. Nhịp bước nhanh mà vẫn gọn. Nhìn tổng thể thể thao, sáng dáng."},
    {"id":"vn_004","tone":"Lãng mạn","shoe_type":"casual","dialogue":"Mình thích vibe đơn giản, nhẹ nhàng mà vẫn gọn. Đi một vòng thấy thoải mái. Tổng thể nhìn tinh tế, dễ hợp nhiều outfit."},
    {"id":"vn_005","tone":"Truyền cảm","shoe_type":"luxury","dialogue":"Mình chọn kiểu tối giản để mọi thứ nhìn sang hơn. Cảm giác bước đi đều, nhẹ. Tổng thể sạch, nhìn có gu."},
]

BUILTIN_SCENES = [
    {"id":"scn_001","shoe_type":"leather","location":"bright boutique studio, marble counter","lighting":"soft daylight, HDR+","motion":"smooth orbit, gentle push-in","weather":"indoor clean","mood":"premium"},
    {"id":"scn_002","shoe_type":"sneaker","location":"daylight cafe minimal, wooden table","lighting":"bright window light, crisp edges","motion":"handheld-stable phone realism, micro sway","weather":"indoor daylight","mood":"fresh"},
    {"id":"scn_003","shoe_type":"runner","location":"modern street morning, clean sidewalk","lighting":"bright morning, controlled highlights","motion":"tracking low angle, smooth glide","weather":"clear sky","mood":"energetic"},
    {"id":"scn_004","shoe_type":"casual","location":"penthouse window light, neutral wall","lighting":"bright window light, no haze","motion":"slow pan + gentle dolly","weather":"indoor","mood":"calm"},
    {"id":"scn_005","shoe_type":"luxury","location":"showroom table macro, premium shelf background","lighting":"high-key lighting, zero noise","motion":"macro move, keep shoe ultra sharp","weather":"indoor","mood":"luxury"},
]

BUILTIN_DISCLAIMERS = [
    "Nội dung chỉ mang tính chia sẻ trải nghiệm.",
    "Video mang tính minh hoạ trải nghiệm thực tế.",
    "Chia sẻ cảm nhận cá nhân, không phải cam kết.",
    "Thông tin trong video chỉ để tham khảo.",
]

# =========================
# LOAD CSV (repo file OR uploaded OR builtin)
# =========================
@st.cache_data
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(pd.io.common.BytesIO(file_bytes), encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(pd.io.common.BytesIO(file_bytes), encoding="utf-8", errors="replace")

@st.cache_data
def read_csv_path(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(p, encoding="utf-8", errors="replace")
        except Exception:
            return None

def df_to_records(df: Optional[pd.DataFrame]) -> List[dict]:
    if df is None or df.empty:
        return []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records")

def load_libraries(uploaded_dialogue, uploaded_scene, uploaded_disclaimer):
    if uploaded_dialogue is not None:
        ddf = read_csv_bytes(uploaded_dialogue.getvalue())
    else:
        ddf = read_csv_path("dialogue_library.csv")

    if uploaded_scene is not None:
        sdf = read_csv_bytes(uploaded_scene.getvalue())
    else:
        sdf = read_csv_path("scene_library.csv")

    if uploaded_disclaimer is not None:
        xdf = read_csv_bytes(uploaded_disclaimer.getvalue())
    else:
        xdf = read_csv_path("disclaimer_prompt2.csv")

    dialogues = df_to_records(ddf) or BUILTIN_DIALOGUES
    scenes = df_to_records(sdf) or BUILTIN_SCENES

    disclaimers = []
    if xdf is not None and not xdf.empty:
        xdf = xdf.copy()
        xdf.columns = [str(c).strip() for c in xdf.columns.tolist()]
        for c in ["disclaimer", "text", "mien_tru", "miễn_trừ", "note", "content", "noi_dung", "line", "script"]:
            if c in xdf.columns:
                arr = xdf[c].dropna().astype(str).tolist()
                disclaimers = [normalize_text(a) for a in arr if normalize_text(a)]
                break
        if not disclaimers:
            col = xdf.columns[-1]
            arr = xdf[col].dropna().astype(str).tolist()
            disclaimers = [normalize_text(a) for a in arr if normalize_text(a)]

    disclaimers = disclaimers or BUILTIN_DISCLAIMERS
    return dialogues, scenes, disclaimers

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

# =========================
# UTILS
# =========================
def pick_unique(pool: List[dict], used_ids: set, key: str) -> dict:
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

def filter_scenes_by_shoe_type(scenes: List[dict], shoe_type: str):
    f = [s for s in scenes if safe_text(s.get("shoe_type")).lower() == shoe_type.lower()]
    return f if f else scenes

def filter_dialogues(dialogues: List[dict], shoe_type: str, tone: str):
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

def pick_n_unique_scenes(scenes: List[dict], shoe_type: str, n: int) -> List[dict]:
    pool = filter_scenes_by_shoe_type(scenes, shoe_type)
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
        picked = next((p for p in preferred if p in available), available[0])

        model = genai.GenerativeModel(picked)

        prompt = (
            "You are a shoe classification system.\n"
            "Return ONLY ONE label from this list:\n"
            f"{', '.join(SHOE_TYPES)}\n\n"
            "Rules:\n"
            "- Return exactly one word.\n"
            "- No explanations.\n"
            "- dress shoe / loafer / oxford / derby => leather.\n"
            "- sports running shoe => runner.\n"
            "- sneaker street => sneaker.\n"
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
# VISUAL STYLE PACK
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

def get_dialogue_3_sentences(row: dict) -> str:
    parts = split_sentences(get_dialogue_from_csv(row))
    a = parts[0] if len(parts) > 0 else "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
    b = parts[1] if len(parts) > 1 else "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định."
    c = parts[2] if len(parts) > 2 else "Nhìn tổng thể tối giản nhưng vẫn có điểm tinh tế."
    return normalize_text(f"{ensure_end_punct(a)}\n{ensure_end_punct(b)}\n{ensure_end_punct(c)}")

def get_dialogue_2_sentences(row: dict) -> str:
    parts = split_sentences(get_dialogue_from_csv(row))
    a = parts[0] if len(parts) > 0 else "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn."
    b = parts[1] if len(parts) > 1 else "Cảm giác di chuyển khá nhẹ nhàng và dễ chịu."
    return normalize_text(f"{ensure_end_punct(a)}\n{ensure_end_punct(b)}")

# =========================
# PROMPT BUILDER (format giống prompt cũ)
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
# SIDEBAR: CSV upload + Gemini key
# =========================
with st.sidebar:
    st.markdown("### Gemini API Key")
    st.caption("Dán key vào đây để AI detect shoe_type. Key chỉ lưu trong session (không hard-code).")

    key_in = st.text_input("Paste your Gemini API key here", value=st.session_state.gemini_api_key, type="password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save key", use_container_width=True):
            st.session_state.gemini_api_key = (key_in or "").strip()
            st.success("Saved for this session.")
    with c2:
        if st.button("Clear key", use_container_width=True):
            st.session_state.gemini_api_key = ""
            st.info("Cleared.")

    st.divider()
    st.markdown("### CSV Libraries (optional)")
    st.caption("Nếu repo đã có 3 file CSV thì không cần upload. Nếu chưa có, upload ở đây để app chạy đúng thư viện của chồng.")
    up_d = st.file_uploader("dialogue_library.csv", type=["csv"], key="u_dialogue")
    up_s = st.file_uploader("scene_library.csv", type=["csv"], key="u_scene")
    up_x = st.file_uploader("disclaimer_prompt2.csv", type=["csv"], key="u_disc")

dialogues, scenes, disclaimers_p2 = load_libraries(up_d, up_s, up_x)

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
    st.write("Mỗi lần bấm **Generate** sẽ ra **5 prompt** và **5 phong cách video** khác nhau (style pack).")
    st.write("Có khóa **sáng + nét** để hạn chế video tối/mờ.")
    st.write("Có khóa **hướng chữ/logo** để tránh bị ngược chữ trên giày.")
    st.caption(f"Loaded dialogues: {len(dialogues)} | scenes: {len(scenes)} | disclaimers: {len(disclaimers_p2)}")

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

    if st.button("Generate", use_container_width=True):
        arr = []
        batch_used_styles = set()

        for _ in range(count):
            style = pick_unique_style_for_batch(batch_used_styles)
            scene_list = pick_n_unique_scenes(scenes, shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)

            d_pool = filter_dialogues(dialogues, shoe_type, tone)
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

            if mode_ui.startswith("PROMPT 1"):
                voice_lines = get_dialogue_3_sentences(d)
                mode = "p1"
            else:
                voice_2 = get_dialogue_2_sentences(d)
                disc_raw = random.choice(disclaimers_p2) if disclaimers_p2 else "Nội dung chỉ mang tính chia sẻ trải nghiệm."
                voice_lines = normalize_text(f"{voice_2}\n{short_disclaimer(disc_raw)}")
                mode = "p2"

            prompt = build_prompt(
                mode=mode,
                shoe_type=shoe_type,
                shoe_name=shoe_name,
                style=style,
                scene_list=scene_list,
                timeline=timeline,
                voice_lines=voice_lines,
                voice_style_line=pick_voice_style(),
            )
            arr.append(prompt)

        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.markdown("### Output prompts")
        tabs = st.tabs([str(i + 1) for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=560, key=f"view_{i}")
                copy_button_unicode_safe(prompts[i], key=f"copy_{i}")
else:
    st.warning("Upload a shoe image to begin.")

st.divider()
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.generated_prompts = []
    st.success("Reset done.")
