import streamlit as st
import pandas as pd
import random
import base64
from pathlib import Path
import re

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sora Prompt Studio Pro – Director Edition",
    layout="wide"
)

st.title("🎬 Sora Prompt Studio Pro – Director Edition")
st.caption("Prompt 1 & 2 • Timeline thoại chuẩn • Không trùng • TikTok Shop SAFE")

# =========================
# CONSTANTS
# =========================
CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

# =========================
# SIDEBAR – GEMINI KEY
# =========================
with st.sidebar:
    st.subheader("🔑 Gemini API Key (tùy chọn)")
    st.caption("Dùng cho AI Vision detect shoe_type")

    if "GEMINI_API_KEY" not in st.session_state:
        st.session_state.GEMINI_API_KEY = ""

    key_input = st.text_input(
        "GEMINI_API_KEY",
        type="password",
        value=st.session_state.GEMINI_API_KEY
    )

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if st.button("💾 Lưu key"):
            st.session_state.GEMINI_API_KEY = key_input.strip()
            st.success("Đã lưu key trong phiên hiện tại.")
    with col_k2:
        if st.button("🗑️ Xóa key"):
            st.session_state.GEMINI_API_KEY = ""
            st.warning("Đã xóa key.")

    if st.session_state.GEMINI_API_KEY:
        st.success("🔐 Key đang hoạt động (session)")
    else:
        st.info("ℹ️ Chưa có Gemini key")

# =========================
# COPY BUTTON
# =========================
def copy_button(text: str, key: str):
    b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    html = f"""
    <button id="{key}" style="
        padding:8px 14px;border-radius:10px;border:1px solid #ccc;
        cursor:pointer;background:#fff;font-weight:600;">📋 COPY</button>
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
            s.innerText = "⚠️ Không copy được";
            setTimeout(()=>s.innerText="",2500);
        }}
    }}
    </script>
    """
    st.components.v1.html(html, height=42)

# =========================
# FILE CHECK
# =========================
required_files = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]
missing = [f for f in required_files if not Path(f).exists()]
if missing:
    st.error(f"❌ Thiếu file: {', '.join(missing)} (phải nằm cùng thư mục app.py)")
    st.stop()

# =========================
# LOAD CSV
# =========================
@st.cache_data
def load_dialogues():
    df = pd.read_csv("dialogue_library.csv")
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv")
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_disclaimer_prompt2():
    df = pd.read_csv("disclaimer_prompt2.csv")
    if "disclaimer" in df.columns:
        arr = df["disclaimer"].dropna().astype(str).tolist()
    else:
        arr = df.iloc[:, -1].dropna().astype(str).tolist()
    return [x.strip() for x in arr if x.strip()]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2()

DISCLAIMER_P1_FALLBACK = [
    "Nội dung chỉ mang tính chia sẻ trải nghiệm cá nhân.",
    "Video mang tính minh họa trải nghiệm, không kêu gọi hành động.",
    "Trải nghiệm có thể khác nhau tùy từng người và điều kiện sử dụng.",
    "Thông tin trong video mang tính tham khảo.",
    "Chi tiết cụ thể vui lòng xem theo từng sản phẩm."
]

# =========================
# MEMORY – CHỐNG TRÙNG
# =========================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []

def pick_unique(pool, used_ids: set, key: str):
    items = [x for x in pool if str(x.get(key, "")).strip() not in used_ids]
    if not items:
        used_ids.clear()
        items = pool[:]
    item = random.choice(items)
    used_ids.add(str(item.get(key, "")).strip())
    return item

# =========================
# UTILS
# =========================
def safe_text(v):
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

def normalize_filename(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[^a-z0-9_]+", " ", n)
    return n

def detect_shoe_heuristic(name: str):
    n = normalize_filename(name)
    if any(x in n for x in ["loafer", "horsebit", "bit", "oxford", "derby"]):
        return "leather"
    if any(x in n for x in ["sandal", "dep"]):
        return "sandals"
    if any(x in n for x in ["boot"]):
        return "boots"
    if any(x in n for x in ["run", "runner", "sport", "thethao"]):
        return "runner"
    if any(x in n for x in ["lux", "premium"]):
        return "luxury"
    if any(x in n for x in ["casual"]):
        return "casual"
    return "sneaker"

def detect_shoe_gemini(image_bytes: bytes, api_key: str):
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "Nhìn vào hình ảnh đôi giày này và trả về CHỈ 1 TỪ trong danh sách: "
            "sneaker, runner, leather, casual, sandals, boots, luxury.\n"
            "Không giải thích, không thêm chữ khác."
        )

        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])

        text = response.text.lower().strip()
        for t in SHOE_TYPES:
            if t in text:
                return t

        return None
    except Exception as e:
        return None

def scene_line(scene):
    return (
        f"{scene.get('lighting','')} • {scene.get('location','')} • "
        f"{scene.get('motion','')} • {scene.get('weather','')} • mood {scene.get('mood','')}"
    ).strip(" •")

def filter_scenes_by_shoe_type(shoe_type):
    f = [s for s in scenes if safe_text(s.get("shoe_type")).lower() == shoe_type.lower()]
    return f if f else scenes

def filter_dialogues(shoe_type, tone):
    tone_f = [d for d in dialogues if safe_text(d.get("tone")) == tone]
    if not tone_f:
        tone_f = dialogues
    shoe_f = [d for d in tone_f if safe_text(d.get("shoe_type")).lower() == shoe_type.lower()]
    return shoe_f if shoe_f else tone_f

def get_dialogue_3_sentences(pool, tone):
    picks = random.sample(pool, k=3) if len(pool) >= 3 else random.choices(pool, k=3)
    lines = []
    for row in picks:
        for col in ["dialogue", "text", "content", "line", "noi_dung"]:
            if col in row:
                t = safe_text(row.get(col))
                if t:
                    lines.append(t)
                    break

    while len(lines) < 3:
        lines.append("Mình thấy cảm giác mang khá tự nhiên và dễ chịu.")

    return " ".join(lines[:3])

# =========================
# BUILD PROMPTS
# =========================
def build_prompt(shoe_type, shoe_name, tone, mode):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

    dialogue_text = get_dialogue_3_sentences(d_pool, tone)
    disclaimer = random.choice(disclaimers_p2) if mode == 2 else random.choice(DISCLAIMER_P1_FALLBACK)

    header = "PROMPT 2 (CÓ CAMEO)" if mode == 2 else "PROMPT 1 (KHÔNG CAMEO)"

    return f"""
SORA VIDEO PROMPT — {header} — TIMELINE LOCK 10s
VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

PRODUCT
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

SCENE
- {scene_line(s)}

AUDIO TIMELINE
0.0–1.2s: Không thoại, ambient + nhạc nền rất nhẹ
1.2–6.9s: VOICE ON (3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.2–6.9s]
{dialogue_text}

SAFETY / MIỄN TRỪ
- {disclaimer}
""".strip()

# =========================
# UI
# =========================
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("📤 Tải ảnh giày", type=["jpg", "png", "jpeg"])

    mode = st.radio("Chọn loại prompt", ["PROMPT 1 – Không cameo", "PROMPT 2 – Có cameo"], index=1)
    tone = st.selectbox("Chọn tone thoại", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    count = st.slider("Số lượng prompt", 1, 10, 5)

with right:
    st.subheader("📌 Hướng dẫn nhanh")
    st.write("1) Upload ảnh • 2) Chọn Prompt 1/2 • 3) Chọn tone • 4) Bấm SINH • 5) Bấm số 1..N để xem & COPY")
    st.caption(f"Dialogues columns: {dialogue_cols}")
    st.caption(f"Scenes columns: {scene_cols}")

st.divider()

if uploaded:
    image_bytes = uploaded.read()
    shoe_name = Path(uploaded.name).stem

    auto_heur = detect_shoe_heuristic(uploaded.name)

    use_ai = st.toggle("🤖 AI Vision detect shoe_type (Gemini)", value=False)

    auto_ai = None
    ai_error = None

    if use_ai and st.session_state.GEMINI_API_KEY:
        auto_ai = detect_shoe_gemini(image_bytes, st.session_state.GEMINI_API_KEY)
        if auto_ai is None:
            ai_error = "Gemini detect lỗi, fallback theo tên file."

    auto_type = auto_ai if auto_ai else auto_heur

    shoe_type_choice = st.selectbox(
        "Chọn shoe_type (Auto / AI / chọn tay)",
        ["Auto"] + SHOE_TYPES,
        index=0
    )

    shoe_type = auto_type if shoe_type_choice == "Auto" else shoe_type_choice

    st.success(f"👟 shoe_type: **{shoe_type}** (Auto: {auto_type})")
    st.info(f"🏷 shoe_name: {shoe_name}")

    if ai_error:
        st.warning(ai_error)

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"

    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            p = build_prompt(
                shoe_type,
                shoe_name,
                tone,
                2 if mode.startswith("PROMPT 2") else 1
            )
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
    st.success("✅ Đã reset")
