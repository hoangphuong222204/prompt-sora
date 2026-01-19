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
    return df.to_dict(orient="records"), [c.strip() for c in df.columns.tolist()]

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv")
    return df.to_dict(orient="records"), [c.strip() for c in df.columns.tolist()]

@st.cache_data
def load_disclaimer_prompt2_flexible():
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
    items = [x for x in pool if str(x.get(key, "")).strip() not in used_ids]
    if not items:
        used_ids.clear()
        items = pool[:]
    item = random.choice(items)
    used_ids.add(str(item.get(key, "")).strip())
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

# =========================
# HEURISTIC DETECT from filename (fallback)
# =========================
def detect_shoe_from_filename(name: str) -> str:
    n = (name or "").lower()
    rules = [
        ("boots",  ["boot", "chelsea", "combat", "martin"]),
        ("sandals",["sandal", "sandals", "dep", "dép", "slipper", "slides"]),
        ("leather",["loafer", "loafers", "moc", "moccasin", "horsebit", "oxford", "derby", "tassel", "brogue", "giaytay", "giày tây", "giay_da", "giayda", "da"]),
        ("runner", ["runner", "running", "jog", "marathon", "gym", "train", "thethao", "thể thao", "sport"]),
        ("casual", ["casual", "daily", "everyday", "basic"]),
        ("luxury", ["lux", "premium", "quietlux", "quiet_lux", "highend", "boutique"]),
        ("sneaker",["sneaker", "sneakers", "kicks", "street"])
    ]
    for shoe_type, keys in rules:
        if any(k in n for k in keys):
            return shoe_type
    return "sneaker"

# =========================
# GEMINI VISION DETECT
# =========================
def gemini_detect_shoe_type(img: Image.Image, api_key: str) -> Optional[str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None

    try:
        import google.generativeai as genai
    except Exception:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Bạn là hệ thống phân loại. Nhìn ảnh giày và chỉ trả về 1 nhãn duy nhất trong danh sách sau:\n"
            f"{', '.join(SHOE_TYPES)}\n\n"
            "Quy tắc:\n"
            "- Trả đúng 1 từ nhãn.\n"
            "- Không giải thích.\n"
            "- Nếu không chắc, chọn nhãn gần nhất.\n"
        )
        resp = model.generate_content([prompt, img])
        text = (resp.text or "").strip().lower()
        text = re.sub(r"[^a-z_]", "", text)
        return text if text in SHOE_TYPES else None
    except Exception:
        return None

# =========================
# DIALOGUE: ensure 3 distinct sentences
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
            "Nhìn tổng thể rất tự nhiên, không bị “gồng”.",
            "Mình thấy hợp những ngày muốn thả lỏng."
        ],
        "close": [
            "Gọn gàng vậy thôi nhưng lại dễ dùng hằng ngày.",
            "Mình thích kiểu đơn giản mà nhìn sạch.",
            "Nhẹ nhàng là đủ đẹp rồi."
        ],
    }
}

def get_dialogue_text(row: dict, tone: str) -> str:
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row:
            t = safe_text(row.get(col))
            if t:
                parts = [p.strip() for p in re.split(r"[.!?]+", t) if p.strip()]
                if len(parts) >= 3:
                    return f"{parts[0]}. {parts[1]}. {parts[2]}."
                if len(parts) == 2:
                    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])
                    extra = random.choice(bank["close"])
                    return f"{parts[0]}. {parts[1]}. {extra}"
                if len(parts) == 1:
                    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])
                    mid = random.choice(bank["mid"])
                    close = random.choice(bank["close"])
                    base = parts[0]
                    if close.lower() in base.lower():
                        close = random.choice([x for x in bank["close"] if x != close])
                    return f"{base}. {mid} {close}"
    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])
    a = random.choice(bank["open"])
    b = random.choice([x for x in bank["mid"] if x != a])
    c = random.choice([x for x in bank["close"] if x != a and x != b])
    return f"{a} {b} {c}"

# =========================
# PROMPTS
# =========================
def build_prompt_p1(shoe_type: str, tone: str, scene: dict, dialogue_text: str, shoe_name: str) -> str:
    # ✅ Prompt 1: KHÔNG cần miễn trừ (đúng ý chồng)
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

def build_prompt_p2(shoe_type: str, tone: str, scene: dict, dialogue_text: str, disclaimer: str, shoe_name: str) -> str:
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

SAFETY / MIỄN TRỪ (PROMPT 2)
- {disclaimer}
""".strip()

# =========================
# SIDEBAR: GEMINI KEY
# =========================
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key (tùy chọn)")
    st.caption("Dùng cho AI Vision detect shoe_type. Không có key vẫn chạy (fallback Auto).")

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
        st.warning("Chưa có key (app vẫn chạy bình thường).")

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
    if Path("disclaimer_prompt1.csv").exists():
        st.success("✅ Đã có disclaimer_prompt1.csv (Prompt 1 sẽ random theo file).")
    else:
        st.info("ℹ️ Prompt 1: KHÔNG cần miễn trừ (đã bỏ).")

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    img = Image.open(uploaded).convert("RGB")

    # ===== Detect (AI + filename) =====
    detected_filename = detect_shoe_from_filename(uploaded.name)

    detected_ai = None
    ai_error = ""
    if st.session_state.gemini_api_key.strip():
        detected_ai = gemini_detect_shoe_type(img, st.session_state.gemini_api_key)
        if detected_ai is None:
            ai_error = "Gemini detect lỗi → fallback Auto theo tên file."
    else:
        ai_error = "Chưa có Gemini key → AI không chạy, fallback Auto theo tên file."

    # ===== Mode selector: Auto / AI / Chọn tay =====
    default_mode = "AI" if detected_ai else "Auto"
    shoe_mode = st.selectbox(
        "Chọn chế độ shoe_type",
        ["AI", "Auto", "Chọn tay"],
        index=["AI", "Auto", "Chọn tay"].index(default_mode),
        help="AI: Gemini Vision | Auto: đoán theo tên file | Chọn tay: bạn tự chọn"
    )

    if shoe_mode == "Chọn tay":
        # mặc định ưu tiên leather nếu có (hay dùng cho giày tây)
        default_idx = SHOE_TYPES.index("leather") if "leather" in SHOE_TYPES else 0
        shoe_type = st.selectbox("Chọn shoe_type (tay)", SHOE_TYPES, index=default_idx)
        st.success(f"👟 Chọn tay: **{shoe_type}**")

    elif shoe_mode == "AI":
        if detected_ai:
            shoe_type = detected_ai
            st.success(f"👟 AI detect shoe_type: **{shoe_type}**")
        else:
            shoe_type = detected_filename
            st.warning(ai_error)
            st.info(f"Fallback Auto (tên file): **{detected_filename}**")

    else:  # Auto
        shoe_type = detected_filename
        st.info(f"👟 Auto theo tên file: **{shoe_type}**")

    st.caption(f"shoe_name (tên file): {shoe_name}")

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            s_pool = filter_scenes_by_shoe_type(shoe_type)
            d_pool = filter_dialogues(shoe_type, tone)

            s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

            dialogue_text = get_dialogue_text(d, tone)

            if mode.startswith("PROMPT 1"):
                p = build_prompt_p1(shoe_type, tone, s, dialogue_text, shoe_name)
            else:
                disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Thông tin trong video mang tính tham khảo."
                p = build_prompt_p2(shoe_type, tone, s, dialogue_text, disclaimer, shoe_name)

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
