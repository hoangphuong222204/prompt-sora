from pathlib import Path

app_code = """import streamlit as st
import pandas as pd
import random
import base64
import re
import requests
from pathlib import Path

# ============================================================
# SORA PROMPT STUDIO PRO — GEMINI EDITION (AI MODE FULL)
# - Luu Gemini API key (button Save/Delete) vao file gemini_key.txt
# - AI mode: (1) doan shoe_type tu ANH (vision) (2) sinh 3 cau thoai khong trung
# - Luon ra 3 cau (CSV neu 1 cau thi lay them 2 cau tu row khac)
# - UI: tab 1..N + nut COPY
# ============================================================

st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
st.title("🎬 Sora Prompt Studio Pro – Director Edition (Gemini AI Mode)")
st.caption("Prompt 1 & 2 • Timeline thoại chuẩn • Không trùng • TikTok Shop SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

KEY_FILE = Path("gemini_key.txt")  # luu cung thu muc app.py

# =========================
# COPY BUTTON (1 CLICK)
# =========================
def copy_button(text: str, key: str):
    b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    html = f\"\"\"
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
    \"\"\"
    st.components.v1.html(html, height=42)

# =========================
# KEY STORAGE
# =========================
def load_saved_key() -> str:
    if KEY_FILE.exists():
        try:
            return KEY_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""

def save_key(k: str) -> bool:
    try:
        KEY_FILE.write_text((k or "").strip(), encoding="utf-8")
        return True
    except Exception:
        return False

def delete_key() -> bool:
    try:
        if KEY_FILE.exists():
            KEY_FILE.unlink()
        return True
    except Exception:
        return False

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = load_saved_key()

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
    df.columns = [c.strip() for c in df.columns]
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv")
    df.columns = [c.strip() for c in df.columns]
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_disclaimer_prompt2_flexible():
    df = pd.read_csv("disclaimer_prompt2.csv")
    df.columns = [c.strip() for c in df.columns]
    cols = df.columns.tolist()
    lower_cols = [c.lower().strip() for c in cols]

    if "disclaimer" in lower_cols:
        c = cols[lower_cols.index("disclaimer")]
        arr = df[c].dropna().astype(str).tolist()
        return [x.strip() for x in arr if x.strip()]

    preferred = ["text", "mien_tru", "miễn_trừ", "note", "content", "noi_dung", "line"]
    for p in preferred:
        if p in lower_cols:
            c = cols[lower_cols.index(p)]
            arr = df[c].dropna().astype(str).tolist()
            return [x.strip() for x in arr if x.strip()]

    if len(cols) >= 2 and lower_cols[0] in ["id", "stt", "no"]:
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
    df.columns = [c.strip() for c in df.columns]
    cols = df.columns.tolist()
    lower_cols = [c.lower() for c in cols]

    if "disclaimer" in lower_cols:
        c = cols[lower_cols.index("disclaimer")]
        arr = df[c].dropna().astype(str).tolist()
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

def pick_unique(pool, used_ids: set, key: str):
    items = [x for x in pool if safe_text(x.get(key, "")).strip() not in used_ids]
    if not items:
        used_ids.clear()
        items = pool[:]
    item = random.choice(items)
    used_ids.add(safe_text(item.get(key, "")).strip())
    return item

# =========================
# DIALOGUE HELPERS (LUÔN 3 CÂU)
# =========================
def split_sentences(text: str):
    t = safe_text(text)
    if not t:
        return []
    parts = re.split(r"[.!?]\\s+", t.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def get_row_text(row):
    for col in ["text", "dialogue", "line", "content", "script", "noi_dung"]:
        if col in row:
            t = safe_text(row.get(col))
            if t:
                return t
    return ""

def build_3_sentences_from_csv(primary_row, tone, pool):
    base = split_sentences(get_row_text(primary_row))
    out = []

    if len(base) >= 1:
        out.append(base[0])
    if len(base) >= 2:
        out.append(base[1])
    if len(base) >= 3:
        out.append(base[2])

    if len(out) < 3:
        candidates = [r for r in pool if safe_text(r.get("id")) != safe_text(primary_row.get("id"))]
        random.shuffle(candidates)
        for r in candidates:
            if len(out) >= 3:
                break
            ss = split_sentences(get_row_text(r))
            if ss:
                out.append(ss[0])

    fallback_by_tone = {
        "Tự tin": [
            "Hôm nay mình đi ra ngoài với nhịp bước gọn gàng.",
            "Nhìn tổng thể dễ phối và cảm giác di chuyển ổn.",
            "Mình thích kiểu đơn giản nhưng vẫn có điểm nhấn."
        ],
        "Truyền cảm": [
            "Có những đôi mang vào là thấy mọi thứ dịu lại.",
            "Mình thích cảm giác vừa vặn và nhìn rất tinh giản.",
            "Càng tối giản, càng dễ tạo phong thái riêng."
        ],
        "Mạnh mẽ": [
            "Mình đi nhanh hơn một chút mà vẫn thấy chắc chân.",
            "Nhịp bước dứt khoát và tổng thể gọn gàng.",
            "Ngày bận rộn thì mình cần sự ổn định như vậy."
        ],
        "Lãng mạn": [
            "Chiều nay ra ngoài chút, tự nhiên mood nhẹ hơn.",
            "Đi chậm thôi nhưng cảm giác lại rất thư thả.",
            "Mình thích sự tinh tế nằm ở những điều giản đơn."
        ],
        "Tự nhiên": [
            "Mình ưu tiên thoải mái, kiểu mang là muốn đi tiếp.",
            "Cảm giác nhẹ nhàng, hợp những ngày muốn thả lỏng.",
            "Nhìn tổng thể rất tự nhiên và đời thường."
        ]
    }
    if tone not in fallback_by_tone:
        tone = "Tự tin"

    while len(out) < 3:
        out.append(random.choice(fallback_by_tone[tone]))

    uniq = []
    for s in out:
        if s not in uniq:
            uniq.append(s)
    while len(uniq) < 3:
        uniq.append(random.choice(fallback_by_tone[tone]))

    return ". ".join(uniq[:3]).rstrip(".") + "."

# =========================
# SHOE AUTO DETECT (TÊN FILE)
# =========================
def normalize_name(s: str):
    s = (s or "").lower()
    s = re.sub(r"[\\W_]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

def detect_shoe_type_from_filename(filename: str):
    n = normalize_name(filename)

    if any(k in n for k in ["loafer", "loafers", "horsebit", "bit", "moc", "mocasin", "moccasin", "oxford", "derby", "monk", "brogue", "dress"]):
        return "leather"
    if any(k in n for k in ["boot", "boots", "chelsea", "chukka"]):
        return "boots"
    if any(k in n for k in ["sandal", "sandals", "dep", "dép", "slipper", "slides"]):
        return "sandals"
    if any(k in n for k in ["runner", "running", "run", "the thao", "thethao", "sport", "gym"]):
        return "runner"
    if any(k in n for k in ["casual", "lifestyle", "everyday", "basic"]):
        return "casual"
    if any(k in n for k in ["lux", "luxury", "premium", "classic", "signature"]):
        return "luxury"
    if any(k in n for k in ["sneaker", "sneakers", "trainer", "trainers"]):
        return "sneaker"

    return "sneaker"

def shoe_name_from_filename(filename: str):
    n = Path(filename).stem
    n = re.sub(r"[_\\-]+", " ", n).strip()
    return n[:60] if n else "shoe"

# =========================
# GEMINI API (TEXT + VISION)
# =========================
def gemini_generate_content(api_key: str, parts, model: str = "gemini-1.5-flash", temperature: float = 0.8, timeout: int = 25):
    if not api_key:
        return None, "NO_KEY"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 512},
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP_{r.status_code}: {r.text[:200]}"
        data = r.json()
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
        return txt, None
    except Exception as e:
        return None, str(e)

def ai_detect_shoe_type_gemini(api_key: str, uploaded_file):
    if not api_key or not uploaded_file:
        return None
    mime = uploaded_file.type or "image/jpeg"
    b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

    prompt = (
        "Chọn đúng 1 nhãn trong danh sách: sneaker, runner, leather, casual, sandals, boots, luxury.\\n"
        "Chỉ trả về đúng 1 từ nhãn (không giải thích).\\n"
        "Gợi ý: leather=giày tây/loafer/oxford/derby/monk; runner=giày chạy; sandals=dép; boots=boot; luxury=dress cao cấp; casual=lifestyle; sneaker=sneaker thường.\\n"
    )

    txt, err = gemini_generate_content(
        api_key,
        parts=[
            {"inline_data": {"mime_type": mime, "data": b64}},
            {"text": prompt},
        ],
        temperature=0.1,
    )
    if err or not txt:
        return None
    out = re.sub(r"[^a-z]", "", txt.strip().lower())
    return out if out in SHOE_TYPES else None

def generate_ai_dialogue_3sent_gemini(api_key: str, shoe_type: str, tone: str, scene_desc: str):
    prompt = f\"\"\"
Viết đúng 3 câu tiếng Việt, đời thường, kiểu chia sẻ trải nghiệm (không quảng cáo trực tiếp).

Bối cảnh: {scene_desc}
Tone: {tone}
Loại giày (chỉ để gợi ý): {shoe_type}

BẮT BUỘC:
- Đúng 3 câu, mỗi câu 7–14 từ.
- Không CTA mua/bán/chốt/đặt hàng.
- Không nói giá/giảm/khuyến mãi.
- Không nói vật liệu nhạy cảm (da, suede, PU...).
- Không so sánh đối thủ.
- Không dùng “cam kết”, “đảm bảo”, “tốt nhất”.
- Viết tự nhiên như nói.
Chỉ trả về 3 câu, không thêm gì khác.
\"\"\"
    txt, err = gemini_generate_content(api_key, parts=[{"text": prompt}], temperature=0.95)
    if err or not txt:
        return None
    sents = split_sentences(txt)
    if len(sents) >= 3:
        return ". ".join(sents[:3]).rstrip(".") + "."
    lines = [x.strip("-• \\t") for x in txt.splitlines() if x.strip()]
    sents2 = []
    for l in lines:
        ss = split_sentences(l)
        if ss:
            sents2.append(ss[0])
    if len(sents2) >= 3:
        return ". ".join(sents2[:3]).rstrip(".") + "."
    return None

# =========================
# FILTER POOLS
# =========================
def filter_scenes_by_shoe_type(shoe_type):
    f = [s for s in scenes if safe_text(s.get("shoe_type")).lower() == shoe_type.lower()]
    return f if f else scenes

def filter_dialogues_by(shoe_type, tone):
    tone_f = [d for d in dialogues if safe_text(d.get("tone")) == tone]
    if not tone_f:
        tone_f = dialogues
    shoe_f = [d for d in tone_f if safe_text(d.get("shoe_type")).lower() == shoe_type.lower()]
    return shoe_f if shoe_f else tone_f

def scene_line(scene):
    return (
        f"{safe_text(scene.get('lighting'))} • {safe_text(scene.get('location'))} • "
        f"{safe_text(scene.get('motion'))} • {safe_text(scene.get('weather'))} • mood {safe_text(scene.get('mood'))}"
    ).strip(" •")

# =========================
# BUILD PROMPTS
# =========================
def build_prompt_p1(shoe_type, shoe_name, tone, ai_mode, api_key):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues_by(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
    disclaimer = random.choice(disclaimers_p1 if disclaimers_p1 else DISCLAIMER_P1_FALLBACK)

    dialogue_text = None
    if ai_mode and api_key:
        dialogue_text = generate_ai_dialogue_3sent_gemini(api_key, shoe_type, tone, scene_line(s))
    if not dialogue_text:
        dialogue_text = build_3_sentences_from_csv(d, tone, d_pool)

    return f\"\"\"
SORA VIDEO PROMPT — PROMPT 1 (KHÔNG CAMEO) — TIMELINE LOCK 10s
VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- KHÔNG người • KHÔNG cameo • KHÔNG xuất hiện nhân vật
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

PRODUCT (REFERENCE)
- shoe_name: {shoe_name}
- shoe_type_hint: {shoe_type} (chỉ để chọn bối cảnh/thoại; Sora ưu tiên ảnh)

SCENE
- {scene_line(s)}

AUDIO TIMELINE
0.0–1.2s: Không thoại, ambient + nhạc nền rất nhẹ
1.2–6.9s: VOICE ON (đúng 3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.2–6.9s]
{dialogue_text}

SAFETY / MIỄN TRỪ
- {disclaimer}
\"\"\".strip()

def build_prompt_p2(shoe_type, shoe_name, tone, ai_mode, api_key):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues_by(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
    disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Thông tin chi tiết vui lòng xem trong giỏ hàng."

    dialogue_text = None
    if ai_mode and api_key:
        dialogue_text = generate_ai_dialogue_3sent_gemini(api_key, shoe_type, tone, scene_line(s))
    if not dialogue_text:
        dialogue_text = build_3_sentences_from_csv(d, tone, d_pool)

    return f\"\"\"
SORA VIDEO PROMPT — PROMPT 2 (CÓ CAMEO) — TIMELINE LOCK 10s
CAMEO VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

CAMEO (FIXED)
- Cameo xuất hiện tự nhiên, review nhẹ nhàng, nói đúng timeline

PRODUCT (REFERENCE)
- shoe_name: {shoe_name}
- shoe_type_hint: {shoe_type} (chỉ để chọn bối cảnh/thoại; Sora ưu tiên ảnh)

SCENE
- {scene_line(s)}

AUDIO TIMELINE
0.0–1.0s: Không thoại, ambient + nhạc nền rất nhẹ
1.0–6.9s: VOICE ON (đúng 3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.0–6.9s]
{dialogue_text}

SAFETY / MIỄN TRỪ (PROMPT 2)
- {disclaimer}
\"\"\".strip()

# =========================
# UI
# =========================
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("📤 Tải ảnh giày", type=["jpg", "png", "jpeg"])

    st.markdown("### 🔑 Gemini API Key")
    key_input = st.text_input("Nhập Gemini API Key", type="password", value=st.session_state.gemini_key)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("💾 Lưu key", use_container_width=True):
            st.session_state.gemini_key = key_input.strip()
            ok = save_key(st.session_state.gemini_key)
            st.success("✅ Đã lưu key" if ok else "⚠️ Không lưu được (quyền thư mục)")
    with c2:
        if st.button("🗑 Xóa key", use_container_width=True):
            ok = delete_key()
            st.session_state.gemini_key = ""
            st.warning("🗑 Đã xóa key" if ok else "⚠️ Không xóa được")
    with c3:
        if st.button("↻ Nạp key đã lưu", use_container_width=True):
            st.session_state.gemini_key = load_saved_key()
            st.info("↻ Đã nạp key")

    api_key = st.session_state.gemini_key.strip()
    if api_key:
        st.success("✅ Key đang hoạt động (đã nhập/lưu)")
    else:
        st.info("ℹ️ Chưa có key: AI MODE sẽ fallback sang CSV.")

    ai_mode = st.checkbox("🤖 AI MODE – Sinh thoại 3 câu (xịn) + (tuỳ chọn) đoán shoe_type từ ảnh", value=False)
    ai_shoe_detect = st.checkbox("🧠 AI đoán shoe_type từ ẢNH (Vision)", value=False, disabled=(not ai_mode))

    mode = st.radio("Chọn loại prompt", ["PROMPT 1 – Không cameo", "PROMPT 2 – Có cameo"], index=1)
    tone = st.selectbox("Chọn tone thoại", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    count = st.slider("Số lượng prompt", 1, 10, 5)

with right:
    st.subheader("📌 Hướng dẫn nhanh")
    st.write("1) Upload ảnh • 2) (Tuỳ chọn) nhập key + bấm Lưu • 3) bật AI MODE • 4) Chọn Prompt 1/2 • 5) Chọn tone • 6) Bấm SINH • 7) Bấm số 1..N để xem & COPY")
    st.caption(f"Dialogues columns: {dialogue_cols}")
    st.caption(f"Scenes columns: {scene_cols}")
    st.caption("Shoe types: " + ", ".join(SHOE_TYPES))

st.divider()

if uploaded:
    shoe_name = shoe_name_from_filename(uploaded.name)
    st.info(f"🪪 shoe_name (lấy từ tên file): **{shoe_name}**")

    auto_type_name = detect_shoe_type_from_filename(uploaded.name)
    auto_type_ai = None

    if ai_mode and ai_shoe_detect and api_key:
        with st.spinner("🤖 AI đang đoán shoe_type từ ảnh..."):
            auto_type_ai = ai_detect_shoe_type_gemini(api_key, uploaded)

    auto_source = "AI ảnh" if auto_type_ai else "Tên file"
    auto_type = auto_type_ai if auto_type_ai else auto_type_name

    shoe_type_choice = st.selectbox("Chọn shoe_type (Auto hoặc chọn tay)", ["Auto"] + SHOE_TYPES, index=0)
    shoe_type = auto_type if shoe_type_choice == "Auto" else shoe_type_choice

    st.success(f"👟 shoe_type: **{shoe_type}** (Auto theo: {auto_source} = {auto_type})")
    if auto_source == "Tên file":
        st.caption("ℹ️ Nếu tên file kiểu image_... thì Auto có thể sai → chọn tay hoặc bật AI Vision.")

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            if mode.startswith("PROMPT 1"):
                p = build_prompt_p1(shoe_type, shoe_name, tone, ai_mode, api_key)
            else:
                p = build_prompt_p2(shoe_type, shoe_name, tone, ai_mode, api_key)
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
"""

out_path = Path("/mnt/data/app.py")
out_path.write_text(app_code, encoding="utf-8")

print(str(out_path))
