import streamlit as st
import pandas as pd
import random
import base64
import re
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
st.title("🎬 Sora Prompt Studio Pro – Director Edition")
st.caption("Prompt 1 & 2 • Tổng 10s • Multi-scene • Timeline thoại chuẩn • Không trùng • TikTok Shop SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"

# shoe_type labels used in your CSV libraries
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

# Minimal files
REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

# =========================
# COPY BUTTON (1 CLICK)
# =========================
def copy_button(text: str, key: str):
    b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    html = f"""
    <button id="{key}" style="
        padding:8px 14px;border-radius:10px;border:1px solid #ccc;
        cursor:pointer;background:#fff;font-weight:700;">📋 COPY</button>
    <span id="{key}_s" style="margin-left:8px;font-size:12px;"></span>
    <script>
    const btn = document.getElementById("{key}");
    const s = document.getElementById("{key}_s");
    btn.onclick = async () => {{
        try {{
            await navigator.clipboard.writeText(atob("{b64}"));
            s.innerText = "✅ Đã copy";
            setTimeout(()=>s.innerText="",1500);
        }} catch(e) {{
            s.innerText = "⚠️ Không copy được (trình duyệt chặn)";
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
    st.error(f"❌ Thiếu file: {', '.join(missing)} (phải nằm cùng thư mục app.py)")
    st.stop()

# =========================
# LOAD CSV
# =========================
@st.cache_data
def load_dialogues():
    df = pd.read_csv("dialogue_library.csv")
    cols = [c.strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records"), cols

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv")
    cols = [c.strip() for c in df.columns.tolist()]
    return df.to_dict(orient="records"), cols

@st.cache_data
def load_disclaimer_prompt2_flexible():
    df = pd.read_csv("disclaimer_prompt2.csv")
    cols = [c.strip() for c in df.columns.tolist()]

    # best: column name "disclaimer"
    if "disclaimer" in cols:
        arr = df["disclaimer"].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    # alternative common column names
    preferred = ["text", "mien_tru", "miễn_trừ", "note", "content", "noi_dung", "line"]
    for c in preferred:
        if c in cols:
            arr = df[c].dropna().astype(str).tolist()
            return [x.strip() for x in arr if x.strip()]

    # if first col is id/stt/no, use second col
    if len(cols) >= 2 and cols[0].lower() in ["id", "stt", "no"]:
        arr = df[cols[1]].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    # fallback last column
    last = cols[-1]
    arr = df[last].dropna().astype(str).tolist()
    return [x.strip() for x in arr if x.strip()]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2_flexible()

# =========================
# SESSION – ANTI DUP
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

def pick_unique(pool, used_ids: set, key: str):
    # allow missing id by generating a stable-ish string key
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

def scene_line(scene: dict) -> str:
    return (
        f"{safe_text(scene.get('lighting'))} • {safe_text(scene.get('location'))} • "
        f"{safe_text(scene.get('motion'))} • {safe_text(scene.get('weather'))} • mood {safe_text(scene.get('mood'))}"
    ).strip(" •")

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
    Total video duration ALWAYS 10.0 seconds.
    2 scenes: 0-5, 5-10
    3 scenes: 0-3.3, 3.3-6.7, 6.7-10
    4 scenes: 0-2.5, 2.5-5, 5-7.5, 7.5-10
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
# FILENAME HEURISTIC (fallback only)
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
    # if filename is just digits => unknown => default "leather"
    if re.fullmatch(r"[\d_\-\.]+", n):
        return "leather"
    return "sneaker"

# =========================
# GEMINI VISION DETECT (AI priority)
# =========================
def gemini_detect_shoe_type(img: Image.Image, api_key: str) -> Tuple[Optional[str], str]:
    """
    Returns (shoe_type or None, raw_text).
    FIXED: auto-pick available Gemini model to avoid 404 NotFound.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "NO_KEY"

    try:
        import google.generativeai as genai
    except Exception as e:
        return None, f"IMPORT_FAIL: {type(e).__name__}: {e}"

    try:
        genai.configure(api_key=api_key)

        # 🔍 List available models first
        models = genai.list_models()
        available = []
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append(m.name)

        if not available:
            return None, "NO_GENERATE_MODELS_AVAILABLE"

        # 🎯 Prefer vision-capable models first
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
            picked = available[0]  # fallback to any usable model

        model = genai.GenerativeModel(picked)

        prompt = (
            "You are a shoe classification system.\n"
            "Return ONLY ONE label from this list:\n"
            f"{', '.join(SHOE_TYPES)}\n\n"
            "Rules:\n"
            "- Return exactly one word.\n"
            "- No explanations.\n"
            "- No extra characters.\n"
            "- If the shoe is a dress shoe, loafer, oxford, derby => leather.\n"
            "- If it is a sports shoe or sneaker => sneaker or runner.\n"
        )

        resp = model.generate_content([prompt, img])
        text = (getattr(resp, "text", "") or "").strip().lower()
        raw = f"{picked} → {text}" if text else f"{picked} → EMPTY_TEXT"

        # normalize
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
            "Gọn – chắc – dễ phối, đúng gu mình.",
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

def get_dialogue_3_sentences(row: dict, tone: str) -> str:
    """
    Prompt 1: Always return exactly 3 sentences (each on new line).
    """
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

    # simple anti-dup
    if b.strip().lower() == a.strip().lower():
        b = random.choice(bank["mid"])
    if c.strip().lower() in {a.strip().lower(), b.strip().lower()}:
        c = random.choice(bank["close"])

    def dot(x):
        x = x.strip()
        return x if x.endswith((".", "!", "?")) else x + "."

    a, b, c = dot(a), dot(b), dot(c)
    return f"{a}\n{b}\n{c}"

def get_dialogue_2_sentences(row: dict, tone: str) -> str:
    """
    Prompt 2: Always return exactly 2 experience sentences (each on new line).
    The 3rd sentence will be the disclaimer.
    """
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

    def dot(x):
        x = x.strip()
        return x if x.endswith((".", "!", "?")) else x + "."

    a, b = dot(a), dot(b)
    return f"{a}\n{b}"

# =========================
# SCRIPT / TABLE + PROMPTS
# =========================
def build_story_overview(shoe_type: str, shoe_name: str) -> str:
    return (
        f"A short 10-second cinematic lifestyle showcase of the men’s shoe ({shoe_name}). "
        f"The video uses clean lighting and subtle camera motion to highlight the shoe’s form and details. "
        f"The tone stays natural and experience-focused (no hard selling). "
        f"Shoe type: {shoe_type}."
    )

def build_scene_table_md(scene_list: List[dict], timeline: List[Tuple[float, float]]) -> str:
    lines = []
    lines.append("| # | Time (within 10s) | Detailed script (VI) | Motion prompt (EN) |")
    lines.append("|---|---|---|---|")

    for i, (sc, (a, b)) in enumerate(zip(scene_list, timeline), start=1):
        vi = (
            f"Bối cảnh: {safe_text(sc.get('location'))}. "
            f"Ánh sáng: {safe_text(sc.get('lighting'))}. "
            f"Chuyển động: {safe_text(sc.get('motion'))}. "
            f"Thời tiết: {safe_text(sc.get('weather'))}. "
            f"Mood: {safe_text(sc.get('mood'))}. "
            "Ưu tiên cận chi tiết giày, chuyển động mượt, cảm giác video thật."
        )

        motion_en = (
            f"Realistic cinematic shot of the shoe in {safe_text(sc.get('location'))}. "
            f"{safe_text(sc.get('lighting'))} lighting. "
            f"Camera movement: {safe_text(sc.get('motion'))}. "
            f"Weather feel: {safe_text(sc.get('weather'))}. "
            f"Mood: {safe_text(sc.get('mood'))}. "
            "Keep the shoe perfectly sharp and unchanged; smooth natural motion; phone-video realism; "
            "NO on-screen text, NO logos, NO watermark."
        )

        lines.append(f"| {i} | {a:.1f}s–{b:.1f}s | {vi} | {motion_en} |")

    return "\n".join(lines)

def build_prompt_unified(
    mode: str,  # "p1" or "p2"
    shoe_type: str,
    shoe_name: str,
    scene_list: List[dict],
    timeline: List[Tuple[float, float]],
    voice_lines: str,      # for p1: 3 lines, for p2: 3 lines (2 + disclaimer)
) -> str:
    story_overview = build_story_overview(shoe_type, shoe_name)
    table_md = build_scene_table_md(scene_list, timeline)

    if mode == "p1":
        title = "PROMPT 1 (NO CAMEO)"
        cast_rule = (
            "CAST RULE\n"
            "- NO people on screen, NO cameo.\n"
            f"- VOICE ID: {CAMEO_VOICE_ID}\n"
        )
    else:
        title = "PROMPT 2 (WITH CAMEO)"
        cast_rule = (
            "CAST RULE\n"
            f"- CAMEO appears naturally like a phone video review (stable, not over-acting).\n"
            f"- CAMEO & VOICE ID: {CAMEO_VOICE_ID}\n"
            "- No hard call-to-action, no price, no discount, no guarantees.\n"
        )

    return f"""
SORA VIDEO PROMPT — {title} — TOTAL 10s LOCK

VIDEO SETUP
- Vertical 9:16, total duration EXACTLY 10 seconds
- Ultra Sharp 4K output
- Realistic video motion (NOT a still image)
- NO on-screen text, NO logos, NO watermark
- NO blur, NO haze, NO glow

{cast_rule}

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- Keep 100% shoe identity: toe shape, panels, stitching, sole, proportions.
- NO redesign, NO deformation, NO guessing, NO color shift.
- LACE RULE: if the reference shoe has laces → keep laces in ALL frames; if NO laces → ABSOLUTELY NO laces.

PRODUCT META
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

STORY OVERVIEW (EN)
{story_overview}

SCENE BREAKDOWN TABLE
{table_md}

AUDIO TIMELINE (TOTAL 10s)
- 0.0–1.2s: no voice, light ambient only
- 1.2–6.9s: voice ON (natural short lines)
- 6.9–10.0s: voice OFF completely, gentle fade-out 9.2–10.0s

VOICEOVER (1.2–6.9s) — {CAMEO_VOICE_ID}
{voice_lines}
""".strip()

# =========================
# SIDEBAR: GEMINI KEY
# =========================
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key (tùy chọn)")
    st.caption("AI Vision detect shoe_type từ ảnh. Nếu AI lỗi sẽ fallback theo tên file.")

    api_key_input = st.text_input("GEMINI_API_KEY", value=st.session_state.gemini_api_key, type="password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Lưu key (phiên này)", use_container_width=True):
            st.session_state.gemini_api_key = api_key_input.strip()
            st.success("✅ Đã lưu key trong phiên hiện tại.")
    with c2:
        if st.button("🗑️ Xóa key", use_container_width=True):
            st.session_state.gemini_api_key = ""
            st.info("Đã xóa key.")

    if st.session_state.gemini_api_key:
        st.success("🔐 Key đang hoạt động (session)")
    else:
        st.warning("Chưa có key (AI sẽ không chạy).")

# =========================
# UI
# =========================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("📤 Tải ảnh giày", type=["jpg", "png", "jpeg"])
    mode = st.radio("Chọn loại prompt", ["PROMPT 1 – Không cameo", "PROMPT 2 – Có cameo"], index=0)
    tone = st.selectbox("Chọn tone thoại", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    scene_count = st.slider("Số phân cảnh (trong tổng 10s)", 2, 4, 3)
    count = st.slider("Số lượng prompt", 1, 10, 5)

with right:
    st.subheader("📌 Hướng dẫn nhanh")
    st.write("1) Upload ảnh • 2) Chọn Prompt 1/2 • 3) Chọn tone • 4) Chọn số phân cảnh • 5) Bấm SINH • 6) COPY")
    st.caption(f"Dialogues columns: {dialogue_cols}")
    st.caption(f"Scenes columns: {scene_cols}")
    st.info("✅ Tổng video: 10s cố định. Multi-scene chia nhỏ trong 10s.")
    st.info("✅ Prompt 1: 3 câu thoại (không miễn trừ).")
    st.info("✅ Prompt 2: 2 câu thoại + 1 câu miễn trừ (tổng 3 câu).")

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

    detect_mode = st.selectbox(
        "Chọn chế độ shoe_type",
        ["AI (ảnh) – ưu tiên", "Auto (tên file) – fallback", "Chọn tay"],
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
            st.success(f"🤖 AI detect shoe_type: **{shoe_type}**")
            st.caption(f"AI raw: {raw_ai}")
        else:
            shoe_type = detected_filename
            st.warning("🤖 AI detect FAILED → fallback theo tên file.")
            st.caption(f"AI raw: {raw_ai}")
            st.info(f"Fallback (tên file): **{detected_filename}**")
    elif detect_mode.startswith("Auto"):
        shoe_type = detected_filename
        st.info(f"🧩 Auto theo tên file: **{shoe_type}**")
    else:
        shoe_type = st.selectbox(
            "Chọn shoe_type thủ công",
            SHOE_TYPES,
            index=SHOE_TYPES.index("leather") if "leather" in SHOE_TYPES else 0
        )
        st.success(f"✅ Chọn tay: **{shoe_type}**")

    st.caption(f"shoe_name (tên file): {shoe_name}")

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            # pick dialogue
            d_pool = filter_dialogues(shoe_type, tone)
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

            # pick multi-scenes inside 10s
            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)

            if mode.startswith("PROMPT 1"):
                # 3 sentences
                voice_lines = get_dialogue_3_sentences(d, tone)
                p = build_prompt_unified("p1", shoe_type, shoe_name, scene_list, timeline, voice_lines)
            else:
                # 2 sentences + 1 disclaimer = 3 total lines
                voice_2 = get_dialogue_2_sentences(d, tone)
                disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Nội dung mang tính chia sẻ trải nghiệm."
                # make disclaimer end with period
                disclaimer = disclaimer.strip()
                if disclaimer and not disclaimer.endswith((".", "!", "?")):
                    disclaimer += "."
                voice_lines = f"{voice_2}\n{disclaimer}"
                p = build_prompt_unified("p2", shoe_type, shoe_name, scene_list, timeline, voice_lines)

            arr.append(p)

        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.markdown("### ✅ Chọn prompt (bấm số)")
        tabs = st.tabs([f"{i+1}" for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=420, key=f"view_{i}")
                copy_button(prompts[i], key=f"copy_view_{i}")

else:
    st.warning("⬆️ Upload ảnh giày để bắt đầu.")

st.divider()
if st.button("♻️ Reset chống trùng"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.generated_prompts = []
    st.success("✅ Đã reset chống trùng.")
