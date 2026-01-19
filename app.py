import streamlit as st
import pandas as pd
import random
import base64
from pathlib import Path

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
st.title("🎬 Sora Prompt Studio Pro – Director Edition")
st.caption("Prompt 1 & 2 • Timeline thoại chuẩn • Không trùng • TikTok Shop SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boot", "luxury"]

# đổi version này mỗi lần update code để phá cache data
CACHE_VERSION = "2026-01-19_v3"

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

def normalize_shoe_type(x: str) -> str:
    s = safe_text(x).lower()
    if s == "boots":
        return "boot"
    return s

def ensure_sentence(s: str) -> str:
    s = safe_text(s)
    if not s:
        return ""
    # đảm bảo có dấu chấm kết câu
    if s[-1] not in ".!?":
        s += "."
    return s

def detect_shoe(name):
    n = (name or "").lower()
    if "loafer" in n or "loafers" in n or "horsebit" in n or "bit" in n:
        return "leather"
    if "oxford" in n or "derby" in n or "dress" in n:
        return "leather"
    if "sandal" in n or "dep" in n:
        return "sandals"
    if "run" in n or "thethao" in n:
        return "runner"
    if "boot" in n or "boots" in n:
        return "boot"
    if "lux" in n:
        return "luxury"
    if "casual" in n:
        return "casual"
    if "da" in n:
        return "leather"
    return "sneaker"

def scene_line(scene):
    return (
        f"{scene.get('lighting','')} • {scene.get('location','')} • "
        f"{scene.get('motion','')} • {scene.get('weather','')} • mood {scene.get('mood','')}"
    ).strip(" •")

# =========================
# LOAD CSV
# =========================
@st.cache_data
def load_dialogues(_v=CACHE_VERSION):
    df = pd.read_csv("dialogue_library.csv")
    if "shoe_type" in df.columns:
        df["shoe_type"] = df["shoe_type"].astype(str).map(normalize_shoe_type)
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_scenes(_v=CACHE_VERSION):
    df = pd.read_csv("scene_library.csv")
    if "shoe_type" in df.columns:
        df["shoe_type"] = df["shoe_type"].astype(str).map(normalize_shoe_type)
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_disclaimer_prompt2_flexible(_v=CACHE_VERSION):
    df = pd.read_csv("disclaimer_prompt2.csv")
    cols = [c.strip() for c in df.columns.tolist()]

    if "disclaimer" in cols:
        arr = df["disclaimer"].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    preferred = ["text", "mien_tru", "miễn_trừ", "note", "content", "noi_dung", "line"]
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

@st.cache_data
def load_disclaimer_prompt1_optional(_v=CACHE_VERSION):
    p = Path("disclaimer_prompt1.csv")
    if not p.exists():
        return None
    df = pd.read_csv(str(p))
    cols = [c.strip() for c in df.columns.tolist()]
    if "disclaimer" in cols:
        arr = df["disclaimer"].dropna().astype(str).tolist()
        arr = [x.strip() for x in arr if x.strip()]
        return arr if arr else None
    last = cols[-1]
    arr = df[last].dropna().astype(str).tolist()
    arr = [x.strip() for x in arr if x.strip()]
    return arr if arr else None

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
if "last_debug" not in st.session_state:
    st.session_state.last_debug = {}

def pick_unique(pool, used_ids: set, key: str):
    items = [x for x in pool if safe_text(x.get(key)) not in used_ids]
    if not items:
        used_ids.clear()
        items = pool[:]
    item = random.choice(items)
    used_ids.add(safe_text(item.get(key)))
    return item

def filter_scenes_by_shoe_type(shoe_type):
    stype = normalize_shoe_type(shoe_type)
    f = [s for s in scenes if normalize_shoe_type(safe_text(s.get("shoe_type"))) == stype]
    return f if f else scenes

def filter_dialogues(shoe_type, tone):
    stype = normalize_shoe_type(shoe_type)
    tone_f = [d for d in dialogues if safe_text(d.get("tone")) == tone]
    if not tone_f:
        tone_f = dialogues
    shoe_f = [d for d in tone_f if normalize_shoe_type(safe_text(d.get("shoe_type"))) == stype]
    return shoe_f if shoe_f else tone_f

def get_one_line(row, tone):
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row.keys():
            t = safe_text(row.get(col))
            if t:
                return ensure_sentence(t)
    # fallback nếu row lạ
    fallback = {
        "Tự tin": [
            "Hôm nay mình đi ra ngoài với nhịp bước gọn gàng hơn.",
            "Nhìn tổng thể dễ phối, cảm giác di chuyển cũng ổn định.",
            "Mình thích kiểu đơn giản nhưng vẫn có điểm nhấn."
        ],
        "Truyền cảm": [
            "Có những đôi mang vào là thấy mọi thứ dịu lại.",
            "Mình thích cảm giác vừa vặn, nhìn kỹ mới thấy cái hay nằm ở sự tinh giản.",
            "Càng tối giản, càng dễ tạo phong cách riêng."
        ],
        "Mạnh mẽ": [
            "Mình đi nhanh hơn một chút mà vẫn thấy chắc chân.",
            "Nhịp bước dứt khoát, gọn gàng, không bị chông chênh.",
            "Ngày bận rộn thì mình cần sự ổn định như vậy."
        ],
        "Lãng mạn": [
            "Chiều nay ra ngoài chút, tự nhiên mood nhẹ hơn.",
            "Đi chậm thôi nhưng cảm giác lại rất thư thả.",
            "Mình thích sự tinh tế nằm ở những thứ giản đơn."
        ],
        "Tự nhiên": [
            "Mình ưu tiên thoải mái, kiểu mang là muốn đi tiếp.",
            "Cảm giác nhẹ nhàng, hợp những ngày muốn thả lỏng.",
            "Nhìn tổng thể rất tự nhiên."
        ]
    }
    return ensure_sentence(random.choice(fallback.get(tone, fallback["Tự tin"])))

def build_dialogue_3_sentences(d_pool, tone):
    """
    ÉP LUÔN 3 câu:
    - bốc 3 dòng khác nhau từ CSV (không trùng id)
    - nếu pool thiếu thì dùng fallback để đủ 3 câu
    """
    k = 3
    lines = []
    local_used = set()
    tries = 0

    while len(lines) < k and tries < 200:
        tries += 1
        d = random.choice(d_pool) if d_pool else {}
        did = safe_text(d.get("id")) or f"fallback_{tries}"
        if did in local_used:
            continue
        local_used.add(did)

        line = get_one_line(d, tone)
        if line:
            lines.append(line)

    while len(lines) < k:
        lines.append(get_one_line({}, tone))

    # ép chắc chắn đủ 3 câu và có dấu kết câu
    lines = [ensure_sentence(x) for x in lines[:3]]
    return " ".join(lines)


# =========================
# BUILD PROMPTS (CÓ KHÓA ẢNH + TÊN GIÀY)
# =========================
def build_prompt(mode, shoe_name, shoe_type, tone, debug=False):
    shoe_type = normalize_shoe_type(shoe_type)
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    dialogue_text = build_dialogue_2_3_sentences(d_pool, tone)

    if mode == "PROMPT 1":
        disclaimer = random.choice(disclaimers_p1 if disclaimers_p1 else DISCLAIMER_P1_FALLBACK)
        header = "SORA VIDEO PROMPT — PROMPT 1 (KHÔNG CAMEO) — TIMELINE LOCK 10s"
        timeline = """AUDIO TIMELINE
0.0–1.2s: Không thoại, ambient + nhạc nền rất nhẹ
1.2–6.9s: VOICE ON (2–3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s
"""
        voice_range = "1.2–6.9s"
        cameo_rule = "- KHÔNG người • KHÔNG cameo • KHÔNG xuất hiện nhân vật"
        disclaimer_title = "SAFETY / MIỄN TRỪ"
    else:
        disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Thông tin trong video mang tính tham khảo."
        header = "SORA VIDEO PROMPT — PROMPT 2 (CÓ CAMEO) — TIMELINE LOCK 10s"
        timeline = """AUDIO TIMELINE
0.0–1.0s: Không thoại, ambient + nhạc nền rất nhẹ
1.0–6.9s: VOICE ON (2–3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s
"""
        voice_range = "1.0–6.9s"
        cameo_rule = "- Có cameo (xuất hiện) • giữ ổn định, không thay người"
        disclaimer_title = "SAFETY / MIỄN TRỪ (PROMPT 2)"

    if debug:
        st.session_state.last_debug = {
            "mode": mode,
            "shoe_name": shoe_name,
            "shoe_type_used": shoe_type,
            "tone_used": tone,
            "dialogue_pool_len": len(d_pool),
            "scene_id": safe_text(s.get("id")),
            "dialogue_preview": dialogue_text[:160]
        }

    return f"""
{header}
VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
{cameo_rule}
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- LOCK 100% shoe identity: toe shape, vamp, panels, stitching, sole, proportions.
- NO redesign • NO deformation • NO guessing • NO color shift.
- LACE RULE: if shoe has laces -> keep laces in ALL frames; if no laces -> ABSOLUTELY NO laces.

PRODUCT
- shoe_name / SKU: {shoe_name}
- shoe_type: {shoe_type}

SCENE
- {scene_line(s)}

{timeline}

[VOICEOVER {CAMEO_VOICE_ID} | {voice_range}]
{dialogue_text}

{disclaimer_title}
- {disclaimer}
""".strip()

# =========================
# UI
# =========================
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("📤 Tải ảnh giày", type=["jpg", "png", "jpeg"])
    mode_ui = st.radio("Chọn loại prompt", ["PROMPT 1 – Không cameo", "PROMPT 2 – Có cameo"], index=1)
    tone = st.selectbox("Chọn tone thoại", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    count = st.slider("Số lượng prompt", 1, 10, 5)
    debug_on = st.checkbox("🛠 DEBUG (xem app đang pick gì)", value=False)

with right:
    st.subheader("📌 Hướng dẫn nhanh")
    st.write("1) Upload ảnh • 2) Nhập TÊN GIÀY/SKU • 3) Chọn shoe_type • 4) Bấm SINH • 5) Bấm số 1..N để xem & COPY")
    st.caption(f"Dialogues columns: {dialogue_cols}")
    st.caption(f"Scenes columns: {scene_cols}")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🔄 Reload thư viện (clear cache)", use_container_width=True):
            st.cache_data.clear()
            st.session_state.generated_prompts = []
            st.session_state.used_dialogue_ids.clear()
            st.session_state.used_scene_ids.clear()
            st.success("✅ Đã clear cache & reset. Bấm SINH lại.")
    with colB:
        st.caption(f"CACHE_VERSION: {CACHE_VERSION}")

st.divider()

if uploaded:
    auto_type = detect_shoe(uploaded.name)
    default_name = Path(uploaded.name).stem

    shoe_name = st.text_input("Tên giày / SKU (rất quan trọng để Sora không đoán sai)", value=default_name).strip()
    if not shoe_name:
        st.error("❌ Chưa nhập Tên giày / SKU.")
        st.stop()

    shoe_type_choice = st.selectbox(
        "Chọn shoe_type (Auto hoặc chọn tay — nên chọn tay cho đúng)",
        ["Auto"] + SHOE_TYPES,
        index=0
    )
    shoe_type = auto_type if shoe_type_choice == "Auto" else shoe_type_choice
    shoe_type = normalize_shoe_type(shoe_type)

    mode = "PROMPT 1" if mode_ui.startswith("PROMPT 1") else "PROMPT 2"
    st.success(f"👟 mode: **{mode}** | shoe_name: **{shoe_name}** | shoe_type: **{shoe_type}** (Auto theo tên file: {auto_type})")

    btn_label = "🎬 SINH PROMPT 1" if mode == "PROMPT 1" else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            arr.append(build_prompt(mode, shoe_name, shoe_type, tone, debug=debug_on))
        st.session_state.generated_prompts = arr

    if debug_on and st.session_state.last_debug:
        st.info(st.session_state.last_debug)

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
