import streamlit as st
import pandas as pd
import random
import base64
from pathlib import Path

st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
st.title("🎬 Sora Prompt Studio Pro – Director Edition")
st.caption("Prompt 1 & 2 • Timeline thoại chuẩn • Không trùng • TikTok Shop SAFE")

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
    # expected: id,tone,shoe_type,text,tags
    return df.to_dict(orient="records")

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv")
    # expected: id,shoe_type,lighting,location,motion,weather,mood
    return df.to_dict(orient="records")

@st.cache_data
def load_disclaimer_prompt2_flexible():
    """
    Hỗ trợ mọi kiểu header:
    - Nếu có cột 'disclaimer' => dùng
    - Nếu không => tự tìm cột text phù hợp:
        ưu tiên: 'text', 'mien_tru', 'mien_tru2', 'note', 'content'
        nếu vẫn không => lấy cột cuối cùng (hoặc cột thứ 2 nếu cột 1 là id)
    """
    df = pd.read_csv("disclaimer_prompt2.csv")
    cols = [c.strip() for c in df.columns.tolist()]

    # 1) chuẩn
    if "disclaimer" in cols:
        arr = df["disclaimer"].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    # 2) thử các tên phổ biến
    preferred = ["text", "mien_tru", "miễn_trừ", "mien_tru2", "note", "content", "noi_dung"]
    for c in preferred:
        if c in cols:
            arr = df[c].dropna().astype(str).tolist()
            return [x.strip() for x in arr if x.strip()]

    # 3) suy luận: nếu có 'id' và có >=2 cột => lấy cột thứ 2
    if len(cols) >= 2 and cols[0].lower() in ["id", "stt", "no"]:
        arr = df[cols[1]].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    # 4) fallback: lấy cột cuối cùng
    last = cols[-1]
    arr = df[last].dropna().astype(str).tolist()
    return [x.strip() for x in arr if x.strip()]

@st.cache_data
def load_disclaimer_prompt1_optional():
    p = Path("disclaimer_prompt1.csv")
    if not p.exists():
        return None
    df = pd.read_csv(str(p))
    cols = [c.strip() for c in df.columns.tolist()]
    if "disclaimer" in cols:
        arr = df["disclaimer"].dropna().astype(str).tolist()
        arr = [x.strip() for x in arr if x.strip()]
        return arr if arr else None
    # fallback: lấy cột cuối
    last = cols[-1]
    arr = df[last].dropna().astype(str).tolist()
    arr = [x.strip() for x in arr if x.strip()]
    return arr if arr else None

dialogues = load_dialogues()
scenes = load_scenes()
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

def pick_unique(pool, used_ids:set, key:str):
    items = [x for x in pool if str(x.get(key, "")).strip() not in used_ids]
    if not items:
        used_ids.clear()
        items = pool[:]
    item = random.choice(items)
    used_ids.add(str(item.get(key, "")).strip())
    return item

# =========================
# SHOE TYPE DETECT
# =========================
def detect_shoe(name):
    n = (name or "").lower()
    if "da" in n: return "leather"
    if "sandal" in n or "dep" in n: return "sandals"
    if "run" in n or "thethao" in n: return "runner"
    if "boot" in n: return "boots"
    if "lux" in n: return "luxury"
    if "casual" in n: return "casual"
    return "sneaker"

def scene_line(scene):
    return f"{scene['lighting']} • {scene['location']} • {scene['motion']} • {scene['weather']} • mood {scene['mood']}"

def filter_scenes_by_shoe_type(shoe_type):
    f = [s for s in scenes if str(s.get("shoe_type","")).strip().lower() == shoe_type.lower()]
    return f if f else scenes

def filter_dialogues(shoe_type, tone):
    tone_f = [d for d in dialogues if str(d.get("tone","")).strip() == tone]
    if not tone_f:
        tone_f = dialogues
    shoe_f = [d for d in tone_f if str(d.get("shoe_type","")).strip().lower() == shoe_type.lower()]
    return shoe_f if shoe_f else tone_f

CAMEO_VOICE_ID = "@phuongnghi18091991"

def build_prompt_p1(shoe_type, tone):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)
    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
    disclaimer = random.choice(disclaimers_p1 if disclaimers_p1 else DISCLAIMER_P1_FALLBACK)

    return f"""
SORA VIDEO PROMPT — PROMPT 1 (KHÔNG CAMEO) — TIMELINE LOCK 10s
VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- KHÔNG người • KHÔNG cameo • KHÔNG xuất hiện nhân vật
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

PRODUCT
- shoe_type: {shoe_type}

SCENE
- {scene_line(s)}

AUDIO TIMELINE
0.0–1.2s: Không thoại, ambient + nhạc nền rất nhẹ
1.2–6.9s: VOICE ON (2–3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.2–6.9s]
{d.get("text","").strip()}

SAFETY / MIỄN TRỪ
- {disclaimer}
""".strip()

def build_prompt_p2(shoe_type, tone):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)
    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
    disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Thông tin chi tiết vui lòng xem trong giỏ hàng."

    return f"""
SORA VIDEO PROMPT — PROMPT 2 (CÓ CAMEO) — TIMELINE LOCK 10s
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
1.0–6.9s: VOICE ON (2–3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.0–6.9s]
{d.get("text","").strip()}

SAFETY / MIỄN TRỪ (PROMPT 2)
- {disclaimer}
""".strip()

# =========================
# UI
# =========================
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("📤 Tải ảnh giày (nhận diện shoe_type theo tên file)", type=["jpg", "png"])
    mode = st.radio("Chọn loại prompt", ["PROMPT 1 – Không cameo", "PROMPT 2 – Có cameo"], index=1)
    tone = st.selectbox("Chọn tone thoại", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    count = st.slider("Số lượng prompt", 1, 10, 5)

with right:
    st.subheader("📌 Hướng dẫn nhanh")
    st.write("1) Upload ảnh  •  2) Chọn Prompt 1/2  •  3) Chọn tone  •  4) Bấm SINH  •  5) COPY dán vào Sora/Veo")
    st.caption("Nếu disclaimer_prompt2.csv header khác, app vẫn tự đọc (đã fix).")

st.divider()

if uploaded:
    shoe_type = detect_shoe(uploaded.name)
    st.success(f"👟 shoe_type nhận diện: **{shoe_type}**")

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        for i in range(count):
            p = build_prompt_p1(shoe_type, tone) if mode.startswith("PROMPT 1") else build_prompt_p2(shoe_type, tone)
            st.markdown(f"### 🎞️ {mode} — #{i+1}")
            st.text_area("Prompt", p, height=360, key=f"prompt_{mode}_{i}")
            copy_button(p, key=f"copy_{mode}_{i}")
else:
    st.warning("⬆️ Upload ảnh giày để bắt đầu tạo prompt.")

st.divider()
if st.button("♻️ Reset chống trùng"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.success("✅ Đã reset bộ nhớ chống trùng")
