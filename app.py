# === FULL APP.PY — AUTO DETECT + PROMPT 1 NO DISCLAIMER ===
import streamlit as st
import pandas as pd
import random
import base64
import re
from pathlib import Path
from typing import Optional
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
st.title("🎬 Sora Prompt Studio Pro – Director Edition")
st.caption("Prompt 1 & 2 • Timeline thoại chuẩn • Không trùng • TikTok Shop SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

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
    return df.to_dict(orient="records")

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv")
    return df.to_dict(orient="records")

@st.cache_data
def load_disclaimer_prompt2():
    df = pd.read_csv("disclaimer_prompt2.csv")
    col = df.columns[-1]
    arr = df[col].dropna().astype(str).tolist()
    return [x.strip() for x in arr if x.strip()]

dialogues = load_dialogues()
scenes = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2()

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

# =========================
# UTILS
# =========================
def safe_text(v):
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except: pass
    s = str(v).strip()
    return "" if s.lower() == "nan" else s

def pick_unique(pool, used_ids, key):
    items = [x for x in pool if str(x.get(key, "")).strip() not in used_ids]
    if not items:
        used_ids.clear()
        items = pool[:]
    item = random.choice(items)
    used_ids.add(str(item.get(key, "")).strip())
    return item

def scene_line(scene):
    return (
        f"{safe_text(scene.get('lighting'))} • {safe_text(scene.get('location'))} • "
        f"{safe_text(scene.get('motion'))} • {safe_text(scene.get('weather'))} • mood {safe_text(scene.get('mood'))}"
    ).strip(" •")

def filter_scenes_by_shoe_type(shoe_type):
    f = [s for s in scenes if safe_text(s.get("shoe_type")).lower() == shoe_type.lower()]
    return f if f else scenes

def filter_dialogues(shoe_type, tone):
    tone_f = [d for d in dialogues if safe_text(d.get("tone")) == tone]
    if not tone_f: tone_f = dialogues
    shoe_f = [d for d in tone_f if safe_text(d.get("shoe_type")).lower() == shoe_type.lower()]
    return shoe_f if shoe_f else tone_f

# =========================
# FILENAME DETECT (STRONGER)
# =========================
def detect_shoe_from_filename(name):
    n = (name or "").lower()
    rules = [
        ("boots",  ["boot", "chelsea", "combat", "martin"]),
        ("sandals",["sandal", "dép", "dep", "slides", "slipper"]),
        ("leather",["loafer", "oxford", "derby", "brogue", "giaytay", "giay_da", "da"]),
        ("runner", ["running", "runner", "gym", "train", "thethao", "sport"]),
        ("luxury", ["lux", "quietlux", "highend", "boutique"]),
        ("casual", ["casual", "daily", "basic"]),
        ("sneaker",["sneaker", "street", "kicks"])
    ]
    for shoe_type, keys in rules:
        if any(k in n for k in keys):
            return shoe_type
    return "sneaker"

# =========================
# GEMINI VISION DETECT
# =========================
def gemini_detect_shoe_type(img, api_key):
    api_key = (api_key or "").strip()
    if not api_key: return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Classify this shoe image. Return ONLY ONE label from:\n"
            f"{', '.join(SHOE_TYPES)}\n"
            "No explanation. One word only."
        )
        resp = model.generate_content([prompt, img])
        text = re.sub(r"[^a-z_]", "", (resp.text or "").lower())
        return text if text in SHOE_TYPES else None
    except:
        return None

# =========================
# PROMPT BUILDERS
# =========================
def build_prompt_p1(shoe_type, tone, scene, dialogue_text, shoe_name):
    return f"""
SORA VIDEO PROMPT — PROMPT 1 (KHÔNG CAMEO) — TIMELINE LOCK 10s
VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- KHÔNG người • KHÔNG cameo • KHÔNG xuất hiện nhân vật
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- KEEP 100% shoe identity (shape, panels, stitching, proportions).
- NO redesign • NO deformation • NO guessing • NO color shift
- If shoe has laces → keep laces in ALL frames. If NO laces → ABSOLUTELY NO laces.

PRODUCT
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

SCENE
- {scene_line(scene)}

AUDIO TIMELINE
0.0–1.2s: Không thoại, ambient + nhạc nền rất nhẹ
1.2–6.9s: VOICE ON (3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.2–6.9s]
{dialogue_text}
""".strip()

def build_prompt_p2(shoe_type, tone, scene, dialogue_text, disclaimer, shoe_name):
    return f"""
SORA VIDEO PROMPT — PROMPT 2 (CÓ CAMEO) — TIMELINE LOCK 10s
CAMEO & VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

CAMEO RULE
- Cameo xuất hiện ổn định, nói tự nhiên như quay điện thoại.
- Không CTA mạnh, không nói giá/khuyến mãi.

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- KEEP 100% shoe identity (shape, panels, stitching, proportions).
- NO redesign • NO deformation • NO guessing • NO color shift
- If shoe has laces → keep laces in ALL frames. If NO laces → ABSOLUTELY NO laces.

PRODUCT
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

SCENE
- {scene_line(scene)}

AUDIO TIMELINE
0.0–1.0s: Không thoại, ambient + nhạc nền rất nhẹ
1.0–6.9s: VOICE ON (3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.0–6.9s]
{dialogue_text}

SAFETY / MIỄN TRỪ
- {disclaimer}
""".strip()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key (tùy chọn)")
    api_key_input = st.text_input("GEMINI_API_KEY", value=st.session_state.gemini_api_key, type="password")
    if st.button("💾 Lưu key"):
        st.session_state.gemini_api_key = api_key_input.strip()
    if st.session_state.gemini_api_key:
        st.success("🔐 Key đang hoạt động")
    else:
        st.info("Chưa có key — dùng filename detect")

# =========================
# UI
# =========================
uploaded = st.file_uploader("📤 Tải ảnh giày", type=["jpg", "png", "jpeg"])
mode = st.radio("Chọn loại prompt", ["PROMPT 1 – Không cameo", "PROMPT 2 – Có cameo"], index=1)
tone = st.selectbox("Chọn tone thoại", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
count = st.slider("Số lượng prompt", 1, 10, 5)

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    img = Image.open(uploaded).convert("RGB")

    ai_type = gemini_detect_shoe_type(img, st.session_state.gemini_api_key)
    file_type = detect_shoe_from_filename(uploaded.name)

    shoe_type = ai_type if ai_type else file_type
    st.success(f"👟 shoe_type sử dụng: **{shoe_type}**")

    if st.button("🎬 SINH PROMPT"):
        arr = []
        for _ in range(count):
            s = pick_unique(filter_scenes_by_shoe_type(shoe_type), st.session_state.used_scene_ids, "id")
            d = pick_unique(filter_dialogues(shoe_type, tone), st.session_state.used_dialogue_ids, "id")

            dialogue_text = safe_text(d.get("dialogue", "")) or "Mình thấy đi khá nhẹ, nhìn tổng thể gọn gàng. Cảm giác di chuyển ổn định, dễ chịu. Tổng thể nhìn đơn giản mà tinh tế."

            if mode.startswith("PROMPT 1"):
                p = build_prompt_p1(shoe_type, tone, s, dialogue_text, shoe_name)
            else:
                disclaimer = random.choice(disclaimers_p2)
                p = build_prompt_p2(shoe_type, tone, s, dialogue_text, disclaimer, shoe_name)

            arr.append(p)

        st.session_state.generated_prompts = arr

    for i, p in enumerate(st.session_state.generated_prompts):
        st.text_area(f"Prompt {i+1}", p, height=420)

