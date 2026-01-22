import streamlit as st
import pandas as pd
import random
import re
import json
import unicodedata
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 & 2 • 10s • Multi-shot • HARD Anti-duplicate • TikTok Shop SAFE • Copy Safe • Style Pack • Key-in-UI • AI Voice Lines")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]
REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

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

def line_signature(line: str) -> str:
    t = compact_spaces(line).lower()
    t = re.sub(r"[“”\"'`]", "", t)
    t = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]

def short_disclaimer(raw: str) -> str:
    s = compact_spaces(normalize_text(raw))
    if not s:
        s = "Nội dung chỉ mang tính chia sẻ trải nghiệm."
    s = ensure_end_punct(s)
    if len(s) > 140:
        s = s[:140].rstrip() + "."
    return normalize_text(s)

def copy_button_unicode_safe(text: str, key: str):
    text = sanitize_for_sora(text)
    payload = json.dumps(text)
    html = f"""
    <button id="{key}" style="padding:8px 14px;border-radius:10px;border:1px solid #ccc;cursor:pointer;background:#fff;font-weight:700;">COPY</button>
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

missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing) + " (must be in same folder as app.py)")
    st.stop()

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
            out = [normalize_text(x) for x in arr if normalize_text(x)]
            return out
    col = df.columns[-1]
    arr = df[col].dropna().astype(str).tolist()
    return [normalize_text(x) for x in arr if normalize_text(x)]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimers()

def _ensure_set(name: str):
    if name not in st.session_state:
        st.session_state[name] = set()

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

_ensure_set("used_dialogue_ids")
_ensure_set("used_scene_ids")
_ensure_set("used_disclaimer_sig")
_ensure_set("used_voice_sig")
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []

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
        preferred = ["models/gemini-2.5-flash","models/gemini-2.0-flash","models/gemini-1.5-flash","models/gemini-1.5-pro","models/gemini-pro-vision","models/gemini-pro"]
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
        return None, f"CALL_FAIL:{type(e).__name__}"

VOICE_STYLE_PACK = [
    "Calm, slow luxury pacing; warm confident male voice; natural pauses; friendly tone",
    "Bright and upbeat but controlled; clear articulation; slight smile in voice; not salesy",
    "Low, deep, confident; steady tempo; short phrases; premium showroom vibe",
    "Soft storytelling; relaxed tempo; gentle emphasis; intimate phone-video feel",
    "Energetic but clean; slightly faster tempo; sporty vibe; keep it natural",
]
def pick_voice_style() -> str:
    return random.choice(VOICE_STYLE_PACK)

STYLE_PACK = [
    {"id":"style_01","name":"Bright boutique studio","lens":"40-50mm","grade":"clean bright luxury","exposure":"bright exposure, HDR+, clean blacks, no underexposure","camera":"smooth orbit and gentle push-in, keep shoe ultra sharp"},
    {"id":"style_02","name":"Daylight cafe minimal","lens":"35-45mm","grade":"soft daylight, crisp edges","exposure":"daylight exposure, highlights controlled, no dim corners","camera":"handheld-stable phone realism, micro sway, no blur"},
    {"id":"style_03","name":"Modern street morning","lens":"28-35mm","grade":"fresh morning contrast","exposure":"bright and clean, no dark shadows, no neon","camera":"tracking low angle, smooth glide, shoe remains sharp"},
    {"id":"style_04","name":"Penthouse window light","lens":"50mm","grade":"premium neutral grade","exposure":"bright window light, high clarity, no haze","camera":"slow pan + gentle dolly, avoid motion blur"},
    {"id":"style_05","name":"Showroom table macro","lens":"60-85mm macro","grade":"ultra sharp product macro","exposure":"high-key lighting, zero blur, zero noise","camera":"macro movement but keep shoe the sharpest object"},
]
def pick_unique_style_for_batch(batch_used: set) -> dict:
    pool = [s for s in STYLE_PACK if s["id"] not in batch_used]
    if not pool:
        batch_used.clear()
        pool = STYLE_PACK[:]
    s = random.choice(pool)
    batch_used.add(s["id"])
    return s

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

DEFAULT_LINES = [
    "Hôm nay mình chọn kiểu gọn gàng để ra ngoài cho tự tin hơn.",
    "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định.",
    "Nhìn tổng thể tối giản nhưng vẫn có điểm tinh tế.",
    "Mình thích cảm giác bước đi gọn, nhìn dáng sáng hơn.",
    "Nhịp di chuyển nhẹ nhàng, tổng thể rất dễ phối đồ.",
]

def reserve_unique_lines(lines: List[str]) -> Optional[List[str]]:
    sigs = [line_signature(ensure_end_punct(x)) for x in lines]
    used = st.session_state.used_voice_sig
    if any(s in used for s in sigs):
        return None
    for s in sigs:
        used.add(s)
    return [ensure_end_punct(x) for x in lines]

def pick_unique_disclaimer() -> str:
    pool = disclaimers_p2[:] if disclaimers_p2 else ["Nội dung chỉ mang tính chia sẻ trải nghiệm."]
    random.shuffle(pool)
    for raw in pool:
        d = short_disclaimer(raw)
        sig = line_signature(d)
        if sig not in st.session_state.used_disclaimer_sig:
            st.session_state.used_disclaimer_sig.add(sig)
            return d
    st.session_state.used_disclaimer_sig.clear()
    d = short_disclaimer(random.choice(pool))
    st.session_state.used_disclaimer_sig.add(line_signature(d))
    return d

def csv_pick_unique_lines(shoe_type: str, tone: str, n_lines: int, max_tries: int = 80) -> List[str]:
    pool = filter_dialogues(shoe_type, tone)
    for _ in range(max_tries):
        row = pick_unique(pool, st.session_state.used_dialogue_ids, "id")
        parts = split_sentences(get_dialogue_from_csv(row))
        lines = []
        for i in range(n_lines):
            if i < len(parts) and parts[i]:
                lines.append(ensure_end_punct(parts[i]))
            else:
                lines.append(DEFAULT_LINES[min(i, len(DEFAULT_LINES)-1)])
        reserved = reserve_unique_lines(lines)
        if reserved:
            return reserved
    out = []
    for cand in DEFAULT_LINES:
        if len(out) >= n_lines:
            break
        if line_signature(ensure_end_punct(cand)) not in st.session_state.used_voice_sig:
            out.append(ensure_end_punct(cand))
    if len(out) < n_lines:
        st.session_state.used_voice_sig.clear()
        out = [ensure_end_punct(x) for x in DEFAULT_LINES[:n_lines]]
        reserve_unique_lines(out)
    return out

def gemini_generate_lines_once(api_key: str, shoe_type: str, tone: str, voice_style: str, n_lines: int) -> Tuple[Optional[List[str]], str]:
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
Viết đúng {n_lines} câu tiếng Việt làm lời thoại review rất ngắn (video 10 giây) về giày.
Chỉ trả về {n_lines} dòng, mỗi dòng 1 câu, không đánh số, không emoji.
Mỗi câu 8–16 từ.
Tone: {tone}.
Cách đọc: {voice_style}.
Nội dung: cảm giác/nhịp bước/độ gọn gàng; tự nhiên như clip điện thoại.
CẤM: giá, giảm giá, khuyến mãi, bảo hành, cam kết tuyệt đối, so sánh, công dụng y tế, vật liệu nhạy cảm.
shoe_type: {shoe_type}
"""
        resp = model.generate_content(prompt)
        text = normalize_text(getattr(resp, "text", "") or "")
        if not text:
            return None, f"{picked}:EMPTY_TEXT"
        lines = [compact_spaces(x) for x in text.split("\n") if compact_spaces(x)]
        lines = lines[:n_lines]
        if len(lines) < n_lines:
            return None, f"{picked}:NOT_ENOUGH_LINES"
        lines = [ensure_end_punct(x) for x in lines]
        return lines, f"{picked}:OK"
    except Exception:
        return None, "CALL_FAIL"

def ai_pick_unique_lines(api_key: str, shoe_type: str, tone: str, voice_style: str, n_lines: int, max_tries: int = 3) -> Tuple[List[str], str]:
    notes = []
    for _ in range(max_tries):
        lines, dbg = gemini_generate_lines_once(api_key, shoe_type, tone, voice_style, n_lines)
        notes.append(dbg)
        if not lines:
            continue
        reserved = reserve_unique_lines(lines)
        if reserved:
            return reserved, "AI_OK(" + ",".join(notes) + ")"
    return csv_pick_unique_lines(shoe_type, tone, n_lines), "AI_FALLBACK(" + ",".join(notes) + ")"

SEP = "══════════════════════════════════"

def build_prompt(mode: str, shoe_type: str, shoe_name: str, style: dict, scene_list: List[dict], timeline: List[Tuple[float,float]], voice_block: str, voice_style_line: str) -> str:
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
        shot_lines.append(ensure_end_punct(f"{a:.1f}–{b:.1f}s: {loc}. {light}. Camera {mot}. Weather {wea}. Mood {mood}."))
    shot_block = "\n".join(shot_lines)

    txt = f"""
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
{sanitize_for_sora(voice_block)}

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
    return sanitize_for_sora(txt)

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

left, right = st.columns([1.05, 0.95])
with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg","png","jpeg"])
    mode_ui = st.radio("Prompt mode", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone = st.selectbox("Tone", ["Truyền cảm","Tự tin","Mạnh mẽ","Lãng mạn","Tự nhiên"], index=1)
    scene_count = st.slider("Shots inside total 10s", 2, 4, 4)
    count = st.slider("Prompts per click", 1, 10, 5)
    detect_mode = st.selectbox("shoe_type detect", ["AI (image) - preferred","Auto (filename) - fallback","Manual"], index=0)

with right:
    st.subheader("Notes")
    st.write("Prompt 2 luôn = **2 câu thoại + 1 câu miễn trừ**.")
    st.write("Không trùng câu thoại giữa Prompt 1 & 2 trong cả session (không đảo câu).")

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_"," ").strip()
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
        dbg_rows = []

        for _ in range(count):
            style = pick_unique_style_for_batch(batch_used_styles)
            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)
            voice_style = pick_voice_style()

            mode = "p1" if mode_ui.startswith("PROMPT 1") else "p2"
            n_lines = 3 if mode == "p1" else 2

            if use_ai_voice and st.session_state.gemini_api_key:
                lines, dbg = ai_pick_unique_lines(st.session_state.gemini_api_key, shoe_type, tone, voice_style, n_lines)
            else:
                lines = csv_pick_unique_lines(shoe_type, tone, n_lines)
                dbg = "CSV_ONLY"

            if mode == "p2":
                disc = pick_unique_disclaimer()
                voice_block = "\n".join(lines + [disc])
            else:
                voice_block = "\n".join(lines)

            arr.append(build_prompt(mode, shoe_type, shoe_name, style, scene_list, timeline, voice_block, voice_style))
            dbg_rows.append({"voice_source": dbg, "style": style.get("id","")})

        st.session_state.generated_prompts = arr
        st.session_state._last_debug = dbg_rows

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.markdown("### Output prompts")
        tabs = st.tabs([str(i+1) for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=620, key=f"view_{i}")
                copy_button_unicode_safe(prompts[i], key=f"copy_{i}")
        if show_ai_debug:
            st.markdown("### Debug (last batch)")
            st.dataframe(pd.DataFrame(st.session_state._last_debug), use_container_width=True)
else:
    st.warning("Upload a shoe image to begin.")

st.divider()
cA, cB, cC = st.columns(3)
with cA:
    if st.button("Reset voice anti-duplicate"):
        st.session_state.used_voice_sig.clear()
        st.success("Reset voice anti-dup done.")
with cB:
    if st.button("Reset scenes/dialogues anti-duplicate"):
        st.session_state.used_scene_ids.clear()
        st.session_state.used_dialogue_ids.clear()
        st.success("Reset scenes/dialogues anti-dup done.")
with cC:
    if st.button("Full reset all anti-duplicate"):
        st.session_state.used_voice_sig.clear()
        st.session_state.used_disclaimer_sig.clear()
        st.session_state.used_scene_ids.clear()
        st.session_state.used_dialogue_ids.clear()
        st.session_state.generated_prompts = []
        st.success("Full reset done.")
