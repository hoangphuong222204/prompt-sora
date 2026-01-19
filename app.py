import streamlit as st
import pandas as pd
import random
import base64
import json
import os
from pathlib import Path

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
st.title("🎬 Sora Prompt Studio Pro – Director Edition")
st.caption("Prompt 1 & 2 • Timeline thoại chuẩn • 3 câu • Không trùng • TikTok Shop SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]
KEY_FILE = Path("gemini_key.json")

# =========================
# GEMINI KEY STORAGE
# =========================
def save_key(k):
    KEY_FILE.write_text(json.dumps({"api_key": k.strip()}), encoding="utf-8")

def load_key():
    if not KEY_FILE.exists():
        return ""
    try:
        return json.loads(KEY_FILE.read_text(encoding="utf-8")).get("api_key", "")
    except Exception:
        return ""

def clear_key():
    if KEY_FILE.exists():
        KEY_FILE.unlink()

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
    cols = [c.strip() for c in df.columns.tolist()]
    if "disclaimer" in cols:
        arr = df["disclaimer"].dropna().astype(str).tolist()
    else:
        arr = df[cols[-1]].dropna().astype(str).tolist()
    return [x.strip() for x in arr if x.strip()]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2()

DISCLAIMER_P1_FALLBACK = [
    "Nội dung chỉ mang tính chia sẻ trải nghiệm cá nhân.",
    "Video mang tính minh họa trải nghiệm, không kêu gọi hành động.",
    "Trải nghiệm có thể khác nhau tùy từng người và điều kiện sử dụng.",
    "Thông tin trong video mang tính tham khảo.",
    "Nội dung không đề cập mua bán, giá hay khuyến mãi."
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
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s

def detect_shoe_by_name(name):
    n = (name or "").lower()
    rules = {
        "leather": ["loafer", "oxford", "derby", "bit", "horsebit", "da"],
        "boots": ["boot"],
        "sandals": ["sandal", "dep"],
        "runner": ["run", "thethao", "sport"],
        "luxury": ["lux", "premium"],
        "casual": ["casual", "daily"],
        "sneaker": ["sneaker", "air", "force", "court"]
    }
    for k, arr in rules.items():
        for kw in arr:
            if kw in n:
                return k
    return "sneaker"

def filter_scenes_by_shoe_type(shoe_type):
    f = [s for s in scenes if safe_text(s.get("shoe_type")).lower() == shoe_type.lower()]
    return f if f else scenes

def filter_dialogues(shoe_type, tone):
    tone_f = [d for d in dialogues if safe_text(d.get("tone")) == tone]
    if not tone_f:
        tone_f = dialogues
    shoe_f = [d for d in tone_f if safe_text(d.get("shoe_type")).lower() == shoe_type.lower()]
    return shoe_f if shoe_f else tone_f

def build_3_sentences(row, tone):
    base = safe_text(row.get("text") or row.get("dialogue") or row.get("content"))
    extras = {
        "Tự tin": [
            "Cảm giác mang rất gọn gàng.",
            "Nhìn tổng thể khá dễ phối đồ.",
            "Di chuyển thấy tự nhiên hơn."
        ],
        "Truyền cảm": [
            "Nhìn kỹ mới thấy cái hay.",
            "Cảm giác mang khá dễ chịu.",
            "Tổng thể nhìn rất nhẹ nhàng."
        ],
        "Mạnh mẽ": [
            "Bước chân chắc và đầm hơn.",
            "Di chuyển thấy ổn định rõ.",
            "Nhịp đi khá vững vàng."
        ],
        "Lãng mạn": [
            "Mood nhìn dịu hơn hẳn.",
            "Cảm giác mang rất thư thả.",
            "Tổng thể nhìn khá tinh tế."
        ],
        "Tự nhiên": [
            "Mang vào thấy rất thoải mái.",
            "Cảm giác khá nhẹ chân.",
            "Đi lại thấy rất tự nhiên."
        ]
    }

    pool = extras.get(tone, extras["Tự tin"])
    random.shuffle(pool)
    if base:
        return f"{base} {pool[0]} {pool[1]}"
    return f"{pool[0]} {pool[1]} {pool[2]}"

def scene_line(scene):
    return (
        f"{scene.get('lighting','')} • {scene.get('location','')} • "
        f"{scene.get('motion','')} • {scene.get('weather','')} • mood {scene.get('mood','')}"
    ).strip(" •")

# =========================
# BUILD PROMPTS
# =========================
def build_prompt(shoe_type, tone, with_cameo=True):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

    dialogue_text = build_3_sentences(d, tone)
    disclaimer = random.choice(disclaimers_p2 if with_cameo else DISCLAIMER_P1_FALLBACK)

    title = "PROMPT 2 (CÓ CAMEO)" if with_cameo else "PROMPT 1 (KHÔNG CAMEO)"

    return f"""
SORA VIDEO PROMPT — {title} — TIMELINE LOCK 10s
CAMEO VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

PRODUCT
- shoe_type: {shoe_type}

SCENE
- {scene_line(s)}

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
# UI
# =========================
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("📤 Tải ảnh giày", type=["jpg", "png", "jpeg"])
    mode = st.radio("Chọn loại prompt", ["PROMPT 1 – Không cameo", "PROMPT 2 – Có cameo"], index=1)
    tone = st.selectbox("Chọn tone thoại", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    count = st.slider("Số lượng prompt", 1, 10, 5)

with right:
    st.subheader("🔑 Gemini AI Mode")
    api_key = st.text_input("Gemini API Key", value=load_key(), type="password")
    colk1, colk2, colk3 = st.columns(3)
    with colk1:
        if st.button("💾 Lưu key"):
            save_key(api_key)
            st.success("Đã lưu key")
    with colk2:
        if st.button("♻️ Nạp key"):
            st.experimental_rerun()
    with colk3:
        if st.button("🗑 Xóa key"):
            clear_key()
            st.warning("Đã xóa key")

    ai_mode = st.checkbox("🤖 AI MODE (Gemini)", value=False)
    vision_mode = st.checkbox("🖼 AI đoán shoe_type từ ẢNH", value=False)

    st.caption(f"Dialogues columns: {dialogue_cols}")
    st.caption(f"Scenes columns: {scene_cols}")

st.divider()

if uploaded:
    auto_type = detect_shoe_by_name(uploaded.name)

    shoe_type_choice = st.selectbox(
        "Chọn shoe_type (Auto hoặc chọn tay)",
        ["Auto"] + SHOE_TYPES,
        index=0
    )

    shoe_type = auto_type if shoe_type_choice == "Auto" else shoe_type_choice
    st.success(f"👟 shoe_type: **{shoe_type}** (Auto đoán: {auto_type})")

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            p = build_prompt(shoe_type, tone, with_cameo=mode.startswith("PROMPT 2"))
            arr.append(p)
        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.markdown("### ✅ Chọn prompt (bấm số)")
        tabs = st.tabs([f"{i+1}" for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=380, key=f"view_{i}")
                copy_button(prompts[i], key=f"copy_view_{i}")

else:
    st.warning("⬆️ Upload ảnh giày để bắt đầu.")

st.divider()
if st.button("♻️ Reset chống trùng"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.generated_prompts = []
    st.success("✅ Đã reset")
