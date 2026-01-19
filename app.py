import streamlit as st
import pandas as pd
import random
import base64
from pathlib import Path
import re

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
st.title("🎬 Sora Prompt Studio Pro – Director Edition")
st.caption("Prompt 1 & 2 • Timeline thoại chuẩn • Không trùng • TikTok Shop SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

# =========================
# COPY BUTTON (1 CLICK)
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
def load_disclaimer_prompt2_flexible():
    df = pd.read_csv("disclaimer_prompt2.csv")
    cols = [c.strip().lower() for c in df.columns.tolist()]

    if "disclaimer" in cols:
        arr = df.iloc[:, cols.index("disclaimer")].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    preferred = ["text", "mien_tru", "miễn_trừ", "note", "content", "noi_dung", "line"]
    for c in preferred:
        if c in cols:
            arr = df.iloc[:, cols.index(c)].dropna().astype(str).tolist()
            return [x.strip() for x in arr if x.strip()]

    if len(cols) >= 2 and cols[0] in ["id", "stt", "no"]:
        arr = df.iloc[:, 1].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    arr = df.iloc[:, -1].dropna().astype(str).tolist()
    return [x.strip() for x in arr if x.strip()]

@st.cache_data
def load_disclaimer_prompt1_optional():
    p = Path("disclaimer_prompt1.csv")
    if not p.exists():
        return None
    df = pd.read_csv(str(p))
    cols = [c.strip().lower() for c in df.columns.tolist()]
    if "disclaimer" in cols:
        arr = df.iloc[:, cols.index("disclaimer")].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]
    arr = df.iloc[:, -1].dropna().astype(str).tolist()
    return [x.strip() for x in arr if x.strip()]

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2_flexible()
disclaimers_p1 = load_disclaimer_prompt1_optional()

DISCLAIMER_P1_FALLBACK = [
    "Nội dung chỉ mang tính chia sẻ trải nghiệm cá nhân.",
    "Video mang tính minh họa trải nghiệm, không kêu gọi hành động.",
    "Trải nghiệm có thể khác nhau tùy từng người và điều kiện sử dụng.",
    "Thông tin trong video mang tính tham khảo.",
    "Chi tiết cụ thể vui lòng xem theo từng sản phẩm.",
    "Nội dung không đề cập mua bán, giá hay khuyến mãi.",
    "Video ghi lại khoảnh khắc sử dụng thực tế, không cam kết tuyệt đối.",
    "Mỗi mẫu có thông tin riêng, vui lòng tham khảo trang sản phẩm.",
    "Nội dung không so sánh với sản phẩm khác.",
    "Video tập trung trải nghiệm hình ảnh và chuyển động."
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

def split_sentences(text):
    parts = re.split(r"[.!?]\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 6]

def get_dialogue_lines(row, tone):
    base_text = ""
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row:
            base_text = safe_text(row.get(col))
            if base_text:
                break

    sentences = split_sentences(base_text)

    fallback_bank = {
        "Tự tin": [
            "Hôm nay mình đi ra ngoài với nhịp bước gọn gàng hơn.",
            "Nhìn tổng thể dễ phối, cảm giác di chuyển cũng rất ổn.",
            "Mình thích kiểu đơn giản nhưng vẫn có điểm nhấn.",
            "Mang vào thấy tự tin hơn khi ra ngoài.",
            "Cảm giác bước đi khá chắc và thoải mái."
        ],
        "Truyền cảm": [
            "Có những đôi mang vào là thấy mọi thứ dịu lại.",
            "Mình thích cảm giác vừa vặn, nhìn kỹ mới thấy cái hay.",
            "Càng tối giản, càng dễ tạo phong cách riêng.",
            "Đi chậm thôi mà cảm giác rất thư thả.",
            "Mỗi bước đi đều mang một cảm xúc nhẹ nhàng."
        ],
        "Mạnh mẽ": [
            "Mình đi nhanh hơn một chút mà vẫn thấy chắc chân.",
            "Nhịp bước dứt khoát, gọn gàng, không bị chông chênh.",
            "Ngày bận rộn thì mình cần sự ổn định như vậy.",
            "Cảm giác di chuyển rất đầm và vững.",
            "Mỗi bước đi đều có lực rõ ràng."
        ],
        "Lãng mạn": [
            "Chiều nay ra ngoài chút, tự nhiên mood nhẹ hơn.",
            "Đi chậm thôi nhưng cảm giác lại rất thư thả.",
            "Mình thích sự tinh tế nằm ở những thứ giản đơn.",
            "Không gian làm cảm xúc dịu lại.",
            "Nhìn tổng thể thấy rất nhẹ nhàng."
        ],
        "Tự nhiên": [
            "Mình ưu tiên thoải mái, kiểu mang là muốn đi tiếp.",
            "Cảm giác nhẹ nhàng, hợp những ngày muốn thả lỏng.",
            "Nhìn tổng thể rất tự nhiên.",
            "Đi lâu mà không thấy mỏi.",
            "Mỗi bước đi đều rất êm."
        ]
    }

    pool = list(set(sentences)) + fallback_bank.get(tone, [])
    random.shuffle(pool)

    chosen = []
    for s in pool:
        if len(chosen) >= 3:
            break
        if all(s not in x for x in chosen):
            chosen.append(s)

    while len(chosen) < 3:
        chosen.append(random.choice(fallback_bank.get(tone, fallback_bank["Tự tin"])))

    return chosen

def detect_shoe(name):
    n = (name or "").lower()
    rules = [
        ("leather", ["loafer", "horsebit", "oxford", "derby", "moc", "da"]),
        ("boots", ["boot", "chelsea", "combat"]),
        ("runner", ["run", "sport", "thethao"]),
        ("sandals", ["sandal", "dep"]),
        ("luxury", ["lux", "premium"]),
        ("casual", ["casual"]),
        ("sneaker", ["sneaker", "trainer", "giay"])
    ]
    for k, arr in rules:
        if any(x in n for x in arr):
            return k
    return "sneaker"

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

# =========================
# BUILD PROMPTS
# =========================
def build_prompt(shoe_type, tone, has_cameo):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

    disclaimer = random.choice(disclaimers_p2) if has_cameo else random.choice(disclaimers_p1 or DISCLAIMER_P1_FALLBACK)
    lines = get_dialogue_lines(d, tone)

    dialogue_text = " ".join(lines)

    label = "PROMPT 2 (CÓ CAMEO)" if has_cameo else "PROMPT 1 (KHÔNG CAMEO)"

    return f"""
SORA VIDEO PROMPT — {label} — TIMELINE LOCK 10s
CAMEO VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- {"KHÔNG người • KHÔNG cameo • KHÔNG xuất hiện nhân vật" if not has_cameo else "Có cameo xuất hiện tự nhiên"}
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

PRODUCT
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
    st.caption("Shoe types: " + ", ".join([str(s).strip() for s in SHOE_TYPES]))

st.divider()

if uploaded:
    auto_type = detect_shoe(uploaded.name)

    shoe_type_choice = st.selectbox(
        "Chọn shoe_type (Auto hoặc chọn tay)",
        ["Auto"] + SHOE_TYPES,
        index=0
    )
    shoe_type = auto_type if shoe_type_choice == "Auto" else shoe_type_choice

    st.success(f"👟 shoe_type: **{shoe_type}** (Auto đoán: {auto_type})")
    st.info(f"📄 shoe_name: **{uploaded.name}**")

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            p = build_prompt(shoe_type, tone, has_cameo=mode.startswith("PROMPT 2"))
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
