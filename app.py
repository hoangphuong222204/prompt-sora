import streamlit as st
import pandas as pd
import random
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Optional deps
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
st.title("🎬 Sora Prompt Studio Pro – Director Edition")
st.caption("Prompt 1 & 2 • Timeline thoại chuẩn • Không trùng • TikTok Shop SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

SHOE_TYPE_LABEL = {
    "sneaker": "Sneaker / giày thể thao",
    "runner": "Runner / chạy bộ",
    "leather": "Giày tây / loafer / dress",
    "casual": "Casual / đi chơi",
    "sandals": "Sandal / dép",
    "boots": "Boots",
    "luxury": "Luxury / high-end",
}

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
# KEY INPUT (Gemini)
# =========================
with st.sidebar:
    st.subheader("🔑 Gemini API Key (tùy chọn)")
    st.caption("Nếu muốn AI Vision nhận diện shoe_type chuẩn, dán Gemini key ở đây.")
    default_key = ""
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")  # Streamlit Cloud: Settings → Secrets
    except Exception:
        default_key = ""
    gemini_key = st.text_input(
        "GEMINI_API_KEY",
        value=st.session_state.get("gemini_key", default_key),
        type="password",
        placeholder="AIza... (Google AI Studio)",
        help="Mẹo: muốn lưu vĩnh viễn trên Streamlit Cloud → Settings → Secrets → GEMINI_API_KEY='...'"
    )
    colk1, colk2 = st.columns(2)
    with colk1:
        if st.button("💾 Lưu key (phiên này)", use_container_width=True):
            st.session_state["gemini_key"] = gemini_key.strip()
            st.success("Đã lưu key trong phiên hiện tại.")
    with colk2:
        if st.button("🧹 Xóa key", use_container_width=True):
            st.session_state["gemini_key"] = ""
            st.success("Đã xóa key trong phiên hiện tại.")
    st.caption("⚠️ Nút 'Lưu key' chỉ lưu trong phiên (session). Muốn lưu vĩnh viễn: dùng Secrets.")

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
    cols = [c.strip() for c in df.columns.tolist()]
    # normalize columns (strip spaces)
    df.columns = cols
    return df.to_dict(orient="records"), cols

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv")
    cols = [c.strip() for c in df.columns.tolist()]
    df.columns = cols
    return df.to_dict(orient="records"), cols

@st.cache_data
def load_disclaimer_prompt2_flexible():
    """
    Hỗ trợ mọi kiểu header cho disclaimer_prompt2.csv
    - ưu tiên cột 'disclaimer'
    - nếu không có -> thử text/content/note...
    - nếu vẫn không -> nếu cột 1 là id -> lấy cột 2, else lấy cột cuối
    """
    df = pd.read_csv("disclaimer_prompt2.csv")
    cols = [c.strip() for c in df.columns.tolist()]
    df.columns = cols

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
    df.columns = cols
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
# MEMORY – CHỐNG TRÙNG + PROMPTS
# =========================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []

def pick_unique(pool: List[Dict], used_ids: set, key: str) -> Dict:
    items = [x for x in pool if str(x.get(key, "")).strip() and str(x.get(key, "")).strip() not in used_ids]
    if not items:
        used_ids.clear()
        items = [x for x in pool if str(x.get(key, "")).strip()] or pool[:]
    item = random.choice(items)
    used_ids.add(str(item.get(key, "")).strip())
    return item

def pick_unique_many(pool: List[Dict], used_ids: set, key: str, n: int) -> List[Dict]:
    """
    Chọn n dòng KHÁC id để ghép thành 3 câu (đỡ bị 1 câu).
    """
    chosen = []
    for _ in range(n):
        row = pick_unique(pool, used_ids, key)
        chosen.append(row)
    return chosen

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

def normalize_tone(t: str) -> str:
    return safe_text(t).strip()

def get_dialogue_col(row: Dict) -> str:
    # ưu tiên cấu trúc bạn đang dùng: 'dialogue'
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row:
            t = safe_text(row.get(col))
            if t:
                return t
    return ""

def compose_voiceover_3_sentences(d_pool: List[Dict], tone: str) -> str:
    """
    Luôn ra 3 câu.
    - Nếu thư viện đủ: lấy 3 dòng khác nhau.
    - Nếu thiếu: fallback theo tone (3 câu).
    """
    tone = normalize_tone(tone)
    if d_pool:
        rows = pick_unique_many(d_pool, st.session_state.used_dialogue_ids, "id", 3)
        lines = [get_dialogue_col(r) for r in rows]
        lines = [x for x in lines if x]
        # nếu có dòng rỗng -> fallback bù
        while len(lines) < 3:
            lines.append(fallback_one_liner(tone))
        # tránh y hệt trong cùng 1 prompt
        uniq = []
        seen = set()
        for x in lines:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        while len(uniq) < 3:
            uniq.append(fallback_one_liner(tone))
        return " ".join(uniq[:3]).strip()

    return fallback_three_liners(tone)

def fallback_one_liner(tone: str) -> str:
    base = {
        "Tự tin": [
            "Hôm nay mình đi ra ngoài với nhịp bước gọn gàng hơn.",
            "Nhìn tổng thể dễ phối, cảm giác di chuyển cũng ổn định.",
            "Mình thích kiểu đơn giản nhưng vẫn có điểm nhấn.",
            "Đi một vòng ngắn mà thấy mọi thứ khá vừa vặn.",
        ],
        "Truyền cảm": [
            "Có những đôi mang vào là thấy mọi thứ dịu lại.",
            "Mình thích cảm giác vừa vặn, nhìn kỹ mới thấy cái hay nằm ở sự tinh giản.",
            "Càng tối giản, càng dễ tạo phong cách riêng.",
            "Đi chậm thôi, nhưng cảm giác lại thư thả hơn hẳn.",
        ],
        "Mạnh mẽ": [
            "Nhịp bước dứt khoát, gọn gàng, không bị chông chênh.",
            "Đi nhanh một chút vẫn thấy chắc chân.",
            "Ngày bận rộn thì mình cần sự ổn định như vậy.",
            "Cảm giác bám chân tốt, mình tự tin di chuyển hơn.",
        ],
        "Lãng mạn": [
            "Chiều nay ra ngoài chút, tự nhiên mood nhẹ hơn.",
            "Đi chậm thôi nhưng cảm giác lại rất thư thả.",
            "Mình thích sự tinh tế nằm ở những thứ giản đơn.",
            "Ánh sáng lên form nhìn cũng mềm hơn.",
        ],
        "Tự nhiên": [
            "Mình ưu tiên thoải mái, kiểu mang là muốn đi tiếp.",
            "Cảm giác nhẹ nhàng, hợp những ngày muốn thả lỏng.",
            "Nhìn tổng thể rất tự nhiên.",
            "Đi cả buổi mà vẫn thấy dễ chịu.",
        ],
    }
    arr = base.get(tone, base["Tự tin"])
    return random.choice(arr)

def fallback_three_liners(tone: str) -> str:
    # luôn 3 câu
    a = fallback_one_liner(tone)
    b = fallback_one_liner(tone)
    c = fallback_one_liner(tone)
    # tránh trùng
    tries = 0
    while (b == a or c in (a, b)) and tries < 10:
        b = fallback_one_liner(tone)
        c = fallback_one_liner(tone)
        tries += 1
    return " ".join([a, b, c]).strip()

def shoe_name_from_filename(name: str) -> str:
    if not name:
        return "uploaded_shoe"
    base = Path(name).stem
    base = base.replace("_", " ").replace("-", " ").strip()
    base = " ".join(base.split())
    return base[:80]

def detect_shoe_by_filename(name: str) -> str:
    """
    Heuristic tốt hơn (không cần API):
    Ưu tiên loafers/giày tây trước -> tránh nhảy về sneaker.
    """
    n = (name or "").lower()
    # loafers / dress
    if any(k in n for k in ["loafer", "loafers", "horsebit", "bit", "moc", "moccasin", "oxford", "derby", "brogue", "monk", "dress"]):
        return "leather"
    if any(k in n for k in ["giaytay", "giay_tay", "giay-da", "giayda", "da-", "da_"]):
        return "leather"
    # boots
    if any(k in n for k in ["boot", "chelsea", "combat", "chukka"]):
        return "boots"
    # sandals
    if any(k in n for k in ["sandal", "sandals", "dep", "dép", "slide", "flipflop"]):
        return "sandals"
    # runner
    if any(k in n for k in ["runner", "running", "run", "thethao", "the_thao", "sport", "gym", "training"]):
        return "runner"
    # casual
    if any(k in n for k in ["casual", "everyday", "daily"]):
        return "casual"
    # luxury
    if any(k in n for k in ["lux", "luxe", "luxury", "premium", "couture"]):
        return "luxury"
    return "sneaker"

def scene_line(scene: Dict) -> str:
    # đảm bảo có đủ key; nếu thiếu thì dùng get
    return (
        f"{safe_text(scene.get('lighting',''))} • {safe_text(scene.get('location',''))} • "
        f"{safe_text(scene.get('motion',''))} • {safe_text(scene.get('weather',''))} • mood {safe_text(scene.get('mood',''))}"
    ).strip(" •")

def filter_scenes_by_shoe_type(shoe_type: str) -> List[Dict]:
    f = [s for s in scenes if safe_text(s.get("shoe_type")).lower() == shoe_type.lower()]
    return f if f else scenes

def filter_dialogues(shoe_type: str, tone: str) -> List[Dict]:
    tone = normalize_tone(tone)
    # lọc tone
    tone_f = [d for d in dialogues if normalize_tone(d.get("tone")) == tone]
    if not tone_f:
        tone_f = dialogues
    # lọc shoe_type
    shoe_f = [d for d in tone_f if safe_text(d.get("shoe_type")).lower() == shoe_type.lower()]
    return shoe_f if shoe_f else tone_f

# =========================
# GEMINI VISION DETECT (optional)
# =========================
@st.cache_data(show_spinner=False)
def gemini_detect_shoe_type(image_bytes: bytes, api_key: str) -> Tuple[str, str]:
    """
    Return (shoe_type, short_name). shoe_type in SHOE_TYPES.
    """
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("Thiếu thư viện google-generativeai. Hãy thêm vào requirements.txt: google-generativeai") from e

    if Image is None:
        raise RuntimeError("Thiếu Pillow. Hãy thêm vào requirements.txt: pillow")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")  # nhanh + rẻ

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prompt = (
        "Bạn là chuyên gia nhận diện GIÀY. "
        "Hãy nhìn ảnh và chọn 1 loại trong danh sách sau (chỉ 1 từ, chữ thường): "
        "sneaker, runner, leather, casual, sandals, boots, luxury. "
        "Quy tắc: loafer/giày tây/giày da/horsebit/oxford/derby/monk => leather. "
        "Ngoài ra, tạo thêm shoe_name ngắn 2-5 từ tiếng Việt (không brand, không vật liệu). "
        "Trả về đúng JSON dạng: {\"shoe_type\":\"...\",\"shoe_name\":\"...\"} và KHÔNG thêm chữ khác."
    )

    res = model.generate_content([prompt, img])
    text = (getattr(res, "text", "") or "").strip()

    # parse JSON safely
    import json, re
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise RuntimeError(f"Gemini trả về không phải JSON: {text[:200]}")
    data = json.loads(m.group(0))
    stype = str(data.get("shoe_type", "")).strip().lower()
    sname = str(data.get("shoe_name", "")).strip()
    if stype not in SHOE_TYPES:
        stype = "leather" if "loafer" in sname.lower() else "sneaker"
    if not sname:
        sname = "giày"
    return stype, sname

# NOTE: need io after cache function definition
import io

# =========================
# BUILD PROMPTS (IMPORTANT: shoe image reference is absolute)
# =========================
def build_prompt_common_header(prompt_no: int, has_cameo: bool) -> str:
    if prompt_no == 1:
        return (
            "SORA VIDEO PROMPT — PROMPT 1 (KHÔNG CAMEO) — TIMELINE LOCK 10s\n"
            f"VOICE ID: {CAMEO_VOICE_ID}\n"
        )
    return (
        "SORA VIDEO PROMPT — PROMPT 2 (CÓ CAMEO) — TIMELINE LOCK 10s\n"
        f"CAMEO VOICE ID: {CAMEO_VOICE_ID}\n"
    )

def build_prompt_p1(shoe_type: str, shoe_name: str, tone: str) -> str:
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    disclaimer = random.choice(disclaimers_p1 if disclaimers_p1 else DISCLAIMER_P1_FALLBACK)
    voiceover = compose_voiceover_3_sentences(d_pool, tone)

    return f"""
{build_prompt_common_header(1, False)}
VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- KHÔNG người • KHÔNG cameo • KHÔNG xuất hiện nhân vật
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- KEEP 100% shoe identity (shape, sole, panels, stitching, proportions).
- NO redesign • NO deformation • NO guessing • NO color shift

PRODUCT (mô tả để đồng bộ)
- shoe_name: {shoe_name}
- shoe_type_hint: {shoe_type} ({SHOE_TYPE_LABEL.get(shoe_type, shoe_type)})

SCENE
- {scene_line(s)}

AUDIO TIMELINE
0.0–1.2s: Không thoại, ambient + nhạc nền rất nhẹ
1.2–6.9s: VOICE ON (3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.2–6.9s]
{voiceover}

SAFETY / MIỄN TRỪ
- {disclaimer}
""".strip()

def build_prompt_p2(shoe_type: str, shoe_name: str, tone: str) -> str:
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Thông tin chi tiết vui lòng xem trong giỏ hàng."
    voiceover = compose_voiceover_3_sentences(d_pool, tone)

    return f"""
{build_prompt_common_header(2, True)}
VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

CAMEO RULE (PROMPT 2)
- Cameo ngồi tự nhiên, nhìn camera, cầm giày rõ chi tiết (không che form).

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- KEEP 100% shoe identity (shape, sole, panels, stitching, proportions).
- NO redesign • NO deformation • NO guessing • NO color shift

PRODUCT (mô tả để đồng bộ)
- shoe_name: {shoe_name}
- shoe_type_hint: {shoe_type} ({SHOE_TYPE_LABEL.get(shoe_type, shoe_type)})

SCENE
- {scene_line(s)}

AUDIO TIMELINE
0.0–1.0s: Không thoại, ambient + nhạc nền rất nhẹ
1.0–6.9s: VOICE ON (3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.0–6.9s]
{voiceover}

SAFETY / MIỄN TRỪ (PROMPT 2)
- {disclaimer}
""".strip()

# =========================
# UI (GỌN)
# =========================
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("📤 Tải ảnh giày", type=["jpg", "png", "jpeg"])
    mode = st.radio("Chọn loại prompt", ["PROMPT 1 – Không cameo", "PROMPT 2 – Có cameo"], index=1)
    tone = st.selectbox("Chọn tone thoại", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    count = st.slider("Số lượng prompt", 1, 10, 5)

with right:
    st.subheader("📌 Hướng dẫn nhanh")
    st.write("1) Upload ảnh • 2) Chọn Prompt 1/2 • 3) Chọn tone • 4) Bấm SINH • 5) Bấm tab số 1..N để xem & COPY")
    st.caption(f"Dialogues columns: {dialogue_cols}")
    st.caption(f"Scenes columns: {scene_cols}")
    st.caption("Shoe types: " + ", ".join(SHOE_TYPES))
    if Path("disclaimer_prompt1.csv").exists():
        st.success("✅ Đã có disclaimer_prompt1.csv (Prompt 1 sẽ random theo file).")
    else:
        st.info("ℹ️ Chưa có disclaimer_prompt1.csv (Prompt 1 dùng danh sách dự phòng).")

st.divider()

# =========================
# MAIN LOGIC
# =========================
if uploaded:
    shoe_name = shoe_name_from_filename(uploaded.name)

    st.info(f"🧾 shoe_name (từ tên file): **{shoe_name}**")

    # AI mode switch
    ai_mode = st.toggle("🤖 AI Vision detect shoe_type (Gemini)", value=bool(st.session_state.get("gemini_key", default_key)))
    api_key_effective = (st.session_state.get("gemini_key") or default_key or "").strip()

    # detect with AI or heuristic
    auto_type = detect_shoe_by_filename(uploaded.name)
    ai_type = None
    ai_name = None

    if ai_mode:
        if not api_key_effective:
            st.warning("AI mode đang bật nhưng chưa có GEMINI_API_KEY. Hãy dán key ở sidebar hoặc tắt AI mode.")
        else:
            try:
                img_bytes = uploaded.getvalue()
                with st.spinner("Gemini đang nhận diện shoe_type..."):
                    ai_type, ai_name = gemini_detect_shoe_type(img_bytes, api_key_effective)
                st.success(f"✅ Gemini detect: **{ai_type}** — gợi ý tên: **{ai_name}**")
                # nếu Gemini gợi ý tên hay hơn -> dùng
                if ai_name and len(ai_name) >= 2:
                    shoe_name = ai_name
            except Exception as e:
                st.warning(f"⚠️ Gemini detect lỗi, dùng heuristic theo tên file. Lỗi: {e}")

    detected = ai_type or auto_type

    shoe_type_choice = st.selectbox(
        "Chọn shoe_type (Auto / AI / hoặc chọn tay)",
        ["Auto/AI"] + SHOE_TYPES,
        index=0
    )
    shoe_type = detected if shoe_type_choice == "Auto/AI" else shoe_type_choice
    st.success(f"👟 shoe_type dùng: **{shoe_type}** (Auto theo tên file: {auto_type}{' | AI: ' + ai_type if ai_type else ''})")

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            p = build_prompt_p1(shoe_type, shoe_name, tone) if mode.startswith("PROMPT 1") else build_prompt_p2(shoe_type, shoe_name, tone)
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
