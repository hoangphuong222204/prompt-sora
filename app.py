import streamlit as st
import pandas as pd
import random
import base64
import re
import json
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 and 2 - Total 10s - Multi-scene - Voice timeline - Anti-duplicate - TikTok Shop SAFE")

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
            s.innerText = "Copy blocked by browser";
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
    st.error("Missing files: " + ", ".join(missing) + " (must be in the same folder as app.py)")
    st.stop()

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

    # FIX Vietnamese mojibake (UTF-8 read as latin1)
    if "Ã" in s or "Â" in s or "Ä" in s:
        try:
            s2 = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if s2 and s2.count("Ã") < s.count("Ã"):
                s = s2
        except Exception:
            pass

    return s

def dot(x: str) -> str:
    x = safe_text(x)
    if not x:
        return ""
    return x if x.endswith((".", "!", "?")) else x + "."

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

# =========================
# LOAD CSV
# =========================
@st.cache_data
def load_dialogues():
    df = pd.read_csv("dialogue_library.csv", encoding="utf-8-sig")
    cols = [c.strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records"), cols

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv", encoding="utf-8-sig")
    cols = [c.strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records"), cols

@st.cache_data
def load_disclaimer_prompt2_flexible():
    df = pd.read_csv("disclaimer_prompt2.csv", encoding="utf-8-sig")
    cols = [c.strip() for c in df.columns.tolist()]

    if "disclaimer" in cols:
        arr = df["disclaimer"].dropna().astype(str).tolist()
        return [safe_text(x) for x in arr if safe_text(x)]

    preferred = ["text", "mien_tru", "miễn_trừ", "note", "content", "noi_dung", "line"]
    for c in preferred:
        if c in cols:
            arr = df[c].dropna().astype(str).tolist()
            return [safe_text(x) for x in arr if safe_text(x)]

    if len(cols) >= 2 and cols[0].lower() in ["id", "stt", "no"]:
        arr = df[cols[1]].dropna().astype(str).tolist()
        return [safe_text(x) for x in arr if safe_text(x)]

    last = cols[-1]
    arr = df[last].dropna().astype(str).tolist()
    return [safe_text(x) for x in arr if safe_text(x)]

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
# FILTERS / TIMELINE
# =========================
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
# GEMINI VISION DETECT - FIX 404
# =========================
def gemini_detect_shoe_type(img: Image.Image, api_key: str) -> Tuple[Optional[str], str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "NO_KEY"

    try:
        import google.generativeai as genai
    except Exception as e:
        return None, f"IMPORT_FAIL: {type(e).__name__}: {e}"

    try:
        genai.configure(api_key=api_key)

        models = genai.list_models()
        available = []
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append(m.name)

        if not available:
            return None, "NO_GENERATE_MODELS_AVAILABLE"

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
            "- Dress shoe, loafer, oxford, derby -> leather.\n"
            "- Sports shoe or sneaker -> sneaker or runner.\n"
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
# DIALOGUE FALLBACK BANK
# =========================
TONE_BANK = {
    "Tự tin": {
        "open": [
            "Hôm nay mình chọn kiểu gọn gàng để ra ngoài tự tin hơn.",
            "Mình thích cảm giác bước đi nhìn gọn và có nhịp.",
            "Mình ưu tiên tổng thể sạch, dễ phối và nhìn sáng dáng."
        ],
        "mid": [
            "Đi một lúc thấy nhịp bước đều, cảm giác khá ổn định.",
            "Mình thấy form lên chân nhìn gọn, dễ đi suốt ngày.",
            "Cảm giác di chuyển nhẹ nhàng, không bị rối mắt."
        ],
        "close": [
            "Tổng thể đơn giản nhưng có điểm tinh tế riêng.",
            "Mình thích kiểu tối giản để tạo phong cách.",
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
            "Đi chậm thôi nhưng cảm giác rất thư thả.",
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
            "Gọn gàng giúp mình tự tin hơn khi di chuyển."
        ],
        "close": [
            "Tổng thể nhìn khỏe mà vẫn sạch.",
            "Gọn, chắc, dễ phối, đúng gu mình.",
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

def get_dialogue_3_sentences_from_csv_or_bank(row: dict, tone: str) -> str:
    candidate = ""
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row and safe_text(row.get(col)):
            candidate = safe_text(row.get(col))
            break

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

    return f"{dot(a)}\n{dot(b)}\n{dot(c)}"

def get_dialogue_2_sentences_from_csv_or_bank(row: dict, tone: str) -> str:
    candidate = ""
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row and safe_text(row.get(col)):
            candidate = safe_text(row.get(col))
            break

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

    return f"{dot(a)}\n{dot(b)}"

# =========================
# GEMINI FREE - GENERATE DIALOGUE LINES
# =========================
def ai_generate_dialogue_lines(api_key: str, shoe_type: str, tone: str, n_lines: int) -> Optional[List[str]]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None

    try:
        import google.generativeai as genai
    except Exception:
        return None

    try:
        genai.configure(api_key=api_key)

        preferred = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
        models = genai.list_models()
        available = [m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
        if not available:
            return None

        picked = None
        for p in preferred:
            if p in available:
                picked = p
                break
        if not picked:
            picked = available[0]

        model = genai.GenerativeModel(picked)

        want = '{"lines":["...","..."]}' if n_lines == 2 else '{"lines":["...","...","..."]}'
        prompt = (
            "Return JSON only: " + want + "\n"
            "Language: Vietnamese.\n"
            "Style: natural spoken, short review about wearing experience.\n"
            f"Tone: {tone}. Shoe type: {shoe_type}.\n"
            "Rules:\n"
            "- No price.\n"
            "- No discount.\n"
            "- No guarantee.\n"
            "- No hard call to action.\n"
            "- No brand names.\n"
            "- No material names.\n"
            "- Each line 6 to 12 words.\n"
            "- Lines must not repeat each other.\n"
        )

        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()

        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
        lines = data.get("lines")
        if not isinstance(lines, list):
            return None

        out: List[str] = []
        for x in lines:
            s = safe_text(x)
            if not s:
                continue
            out.append(dot(s))

        if len(out) != n_lines:
            return None
        return out

    except Exception:
        return None

# =========================
# COMPACT SCENE LINES (NO MARKDOWN)
# =========================
def build_scene_lines_compact(scene_list: List[dict], timeline: List[Tuple[float, float]]) -> str:
    out = []
    for sc, (a, b) in zip(scene_list, timeline):
        loc = safe_text(sc.get("location"))
        lig = safe_text(sc.get("lighting"))
        mot = safe_text(sc.get("motion"))
        wea = safe_text(sc.get("weather"))
        moo = safe_text(sc.get("mood"))
        line = f"{a:.1f}-{b:.1f}s: location={loc}; lighting={lig}; motion={mot}; weather={wea}; mood={moo}"
        line = re.sub(r"\s{2,}", " ", line).strip()
        out.append(line)
    return "\n".join(out)

# =========================
# PROMPT BUILDER - COMPACT, SORA-SAFE
# =========================
def build_prompt_compact(
    mode: str,  # "p1" or "p2"
    shoe_type: str,
    shoe_name: str,
    scene_list: List[dict],
    timeline: List[Tuple[float, float]],
    voice_lines: str,
) -> str:
    scenes_text = build_scene_lines_compact(scene_list, timeline)

    if mode == "p1":
        title = "PROMPT 1 (NO CAMEO)"
        cast_rule = f"No people on screen. No cameo. Voice ID: {CAMEO_VOICE_ID}."
    else:
        title = "PROMPT 2 (WITH CAMEO)"
        cast_rule = (
            f"Cameo appears naturally like a phone video review. Cameo and Voice ID: {CAMEO_VOICE_ID}. "
            "No hard call to action. No price. No discount. No guarantees."
        )

    return f"""
SORA VIDEO PROMPT - {title} - TOTAL 10s

Video setup:
- Vertical 9:16
- Total duration exactly 10 seconds
- Ultra sharp 4K output
- Realistic motion (not a still image)
- No on-screen text
- No logo
- No watermark
- No blur, haze, or glow

Cast rule:
{cast_rule}

Shoe reference lock:
- Use only the uploaded shoe image as reference
- Keep 100 percent shoe identity: toe shape, panels, stitching, sole, proportions
- No redesign, no deformation, no guessing, no color shift
- Lace rule: if reference shoe has laces then keep laces in all frames; if no laces then absolutely no laces

Product:
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

Scenes (inside 10s):
{scenes_text}

Audio timeline:
- 0.0-1.2s: no voice, light ambient only
- 1.2-6.9s: voice on
- 6.9-10.0s: voice off, gentle fade out 9.2-10.0s

Voiceover (1.2-6.9s):
{voice_lines}
""".strip()

# =========================
# SIDEBAR: GEMINI KEY
# =========================
with st.sidebar:
    st.markdown("### Gemini API Key (optional)")
    st.caption("Used for AI shoe_type detection and AI dialogue generation. If AI fails, fallback is used.")
    api_key_input = st.text_input("GEMINI_API_KEY", value=st.session_state.gemini_api_key, type="password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save key (this session)", use_container_width=True):
            st.session_state.gemini_api_key = api_key_input.strip()
            st.success("Saved.")
    with c2:
        if st.button("Clear key", use_container_width=True):
            st.session_state.gemini_api_key = ""
            st.info("Cleared.")

    if st.session_state.gemini_api_key:
        st.success("Key is set (session).")
    else:
        st.warning("No key. AI features will be disabled.")

# =========================
# UI
# =========================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
    mode = st.radio("Prompt type", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone = st.selectbox("Voice tone", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    scene_count = st.slider("Number of scenes (within total 10s)", 2, 4, 3)
    count = st.slider("Number of prompts", 1, 10, 5)
    use_ai_dialogue = st.checkbox("Prefer AI dialogue (Gemini)", value=True)

with right:
    st.subheader("Notes")
    st.write("Total duration is always 10 seconds. Scenes are split inside 10 seconds.")
    st.write("Prompt 1: 3 voice lines. No disclaimer line.")
    st.write("Prompt 2: 2 voice lines plus 1 short disclaimer line (total 3 lines).")
    st.caption("Dialogues columns: " + ", ".join(dialogue_cols))
    st.caption("Scenes columns: " + ", ".join(scene_cols))

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

    detect_mode = st.selectbox(
        "shoe_type mode",
        ["AI (image) - preferred", "Auto (filename) - fallback", "Manual pick"],
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
        for _ in range(count):
            d_pool = filter_dialogues(shoe_type, tone)
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)

            if mode.startswith("PROMPT 1"):
                ai_lines = ai_generate_dialogue_lines(st.session_state.gemini_api_key, shoe_type, tone, n_lines=3) if use_ai_dialogue else None
                if ai_lines:
                    voice_lines = "\n".join(ai_lines)
                else:
                    voice_lines = get_dialogue_3_sentences_from_csv_or_bank(d, tone)

                p = build_prompt_compact("p1", shoe_type, shoe_name, scene_list, timeline, voice_lines)

            else:
                ai_lines = ai_generate_dialogue_lines(st.session_state.gemini_api_key, shoe_type, tone, n_lines=2) if use_ai_dialogue else None
                if ai_lines:
                    voice_2 = "\n".join(ai_lines)
                else:
                    voice_2 = get_dialogue_2_sentences_from_csv_or_bank(d, tone)

                disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Nội dung mang tính chia sẻ trải nghiệm."
                disclaimer = dot(disclaimer)

                voice_lines = f"{voice_2}\n{disclaimer}"

                p = build_prompt_compact("p2", shoe_type, shoe_name, scene_list, timeline, voice_lines)

            arr.append(p)

        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.markdown("### Output prompts")
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
