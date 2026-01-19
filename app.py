import streamlit as st
import pandas as pd
import random
import base64
from pathlib import Path
from typing import List, Optional
import re
import os

# AI (Gemini)
try:
    import google.generativeai as genai
    GEMINI_OK = True
except Exception:
    GEMINI_OK = False

# Image utils
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False


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
    df.columns = [c.strip() for c in df.columns]
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_scenes():
    df = pd.read_csv("scene_library.csv")
    df.columns = [c.strip() for c in df.columns]
    return df.to_dict(orient="records"), df.columns.tolist()

@st.cache_data
def load_disclaimer_prompt2_flexible():
    """
    Hỗ trợ mọi kiểu header cho disclaimer_prompt2.csv
    - ưu tiên cột 'disclaimer'
    - nếu không có -> thử text/content/note...
    - nếu vẫn không -> nếu cột 1 là id -> lấy cột 2, else lấy cột cuối
    """
    df = pd.read_csv("disclaimer_prompt2.csv")
    df.columns = [c.strip() for c in df.columns]
    cols = df.columns.tolist()

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
    df.columns = [c.strip() for c in df.columns]
    cols = df.columns.tolist()
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

def normalize_filename_to_shoename(name: str) -> str:
    if not name:
        return "shoe"
    # bỏ đuôi
    base = re.sub(r"\.(jpg|jpeg|png|webp|bmp)$", "", name.strip(), flags=re.I)
    base = re.sub(r"[_\-]+", " ", base).strip()
    base = re.sub(r"\s+", " ", base)
    return base[:80] if base else "shoe"

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

def get_dialogue_column_value(row):
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row:
            t = safe_text(row.get(col))
            if t:
                return t
    return ""

def get_3_lines_from_csv(d_pool, tone: str) -> str:
    """
    Fallback không AI: lấy 3 dòng khác nhau (3 id khác nhau) để tránh 1 câu.
    Nếu không đủ -> dùng fallback tone để bù.
    """
    chosen = []
    tmp_used = set()

    # lấy tối đa 3 dòng khác nhau
    for _ in range(20):
        if len(chosen) >= 3:
            break
        d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
        did = str(d.get("id", "")).strip()
        if did in tmp_used:
            continue
        line = get_dialogue_column_value(d)
        if line:
            chosen.append(line)
            tmp_used.add(did)

    fallback = {
        "Tự tin": [
            "Hôm nay mình giữ nhịp bước gọn gàng và tự nhiên hơn.",
            "Tổng thể nhìn dễ phối, cảm giác di chuyển cũng ổn định.",
            "Mình thích kiểu đơn giản nhưng vẫn có điểm nhấn."
        ],
        "Truyền cảm": [
            "Có những lúc chỉ cần bước chậm lại là thấy mọi thứ dịu hơn.",
            "Mình thích cảm giác vừa vặn, nhìn kỹ mới thấy cái hay nằm ở sự tinh giản.",
            "Càng tối giản, càng dễ tạo phong thái riêng."
        ],
        "Mạnh mẽ": [
            "Mình đi nhanh hơn một chút mà vẫn thấy chắc chân.",
            "Nhịp bước dứt khoát, gọn gàng, không bị chông chênh.",
            "Ngày bận rộn thì mình ưu tiên sự ổn định như vậy."
        ],
        "Lãng mạn": [
            "Chiều nay ra ngoài một chút, tự nhiên mood nhẹ hơn.",
            "Đi chậm thôi nhưng cảm giác lại rất thư thả.",
            "Mình thích sự tinh tế nằm ở những thứ giản đơn."
        ],
        "Tự nhiên": [
            "Mình ưu tiên thoải mái, kiểu mang là muốn đi tiếp.",
            "Cảm giác nhẹ nhàng, hợp những ngày muốn thả lỏng.",
            "Nhìn tổng thể rất tự nhiên."
        ]
    }
    fb = fallback.get(tone, fallback["Tự tin"])

    # bù cho đủ 3 câu
    while len(chosen) < 3:
        chosen.append(random.choice(fb))

    # làm sạch + ghép
    chosen = [re.sub(r"\s+", " ", x).strip() for x in chosen]
    return " ".join(chosen[:3])


# =========================
# GEMINI AI MODE
# =========================
def gemini_configure(api_key: str) -> bool:
    if not api_key:
        return False
    if not GEMINI_OK:
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception:
        return False

def gemini_generate_3_sentences(api_key: str, shoe_type: str, tone: str, scene_hint: str, shoe_name: str) -> Optional[str]:
    """
    Sinh đúng 3 câu tiếng Việt, TikTok-safe, không CTA, không giá/khuyến mãi,
    không nhắc vật liệu nhạy cảm, không brand, không cam kết tuyệt đối.
    """
    if not gemini_configure(api_key):
        return None

    model_name = "gemini-1.5-flash"
    model = genai.GenerativeModel(model_name)

    # tăng đa dạng: nhiệt độ + random seed tự nhiên
    temp = random.choice([0.9, 1.0, 1.1, 1.2])

    prompt = f"""
Bạn là người viết lời thoại review đời thường cho video giày (TikTok Shop SAFE).
Hãy viết CHÍNH XÁC 3 câu tiếng Việt (mỗi câu 8–16 từ), văn nói tự nhiên.

Ràng buộc bắt buộc:
- CHỈ 3 câu, ngăn cách bằng dấu " | " (pipe).
- Không kêu gọi mua, không CTA, không "mua/bán/chốt/ib/inbox/link".
- Không giá, không khuyến mãi, không cam kết tuyệt đối (không "tốt nhất/đảm bảo/100%").
- Không so sánh đối thủ, không nhắc thương hiệu.
- Không nhắc vật liệu nhạy cảm (da bò/da lợn/suede/PU...).
- Nội dung là chia sẻ cảm nhận khi di chuyển: êm, chắc, gọn, dễ phối, ổn định...
- Tone: {tone}
- Shoe type: {shoe_type}
- Gợi ý bối cảnh: {scene_hint}
- Tên nội bộ đôi giày: {shoe_name} (chỉ dùng để gợi ý, không cần nhắc lại)

Xuất đúng định dạng:
câu1 | câu2 | câu3
""".strip()

    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": temp,
                "top_p": 0.95,
                "max_output_tokens": 120
            }
        )
        text = (resp.text or "").strip()
        if not text:
            return None

        # parse 3 câu bằng |
        parts = [re.sub(r"\s+", " ", p).strip(" .") for p in text.split("|")]
        parts = [p for p in parts if p]

        # nếu model lỡ xuống dòng / đánh số -> cố cứu
        if len(parts) < 3:
            lines = [re.sub(r"^\d+[\)\.\-]\s*", "", x.strip()) for x in re.split(r"[\n\r]+", text) if x.strip()]
            # gom lại, lấy 3 dòng đầu
            parts = lines[:3]

        # đảm bảo đúng 3 câu
        if len(parts) < 3:
            return None
        parts = parts[:3]
        # thêm dấu chấm cuối câu
        parts = [p + "." if not p.endswith((".", "!", "?")) else p for p in parts]
        return " ".join(parts)

    except Exception:
        return None

def gemini_detect_shoe_type_from_image(api_key: str, image_bytes: bytes) -> Optional[str]:
    """
    Đoán shoe_type từ ảnh: chỉ trả về 1 trong SHOE_TYPES.
    """
    if not gemini_configure(api_key):
        return None
    if not PIL_OK:
        return None
    try:
        img = Image.open(Path("tmp_upload.png"))  # fallback nếu có file
    except Exception:
        try:
            from io import BytesIO
            img = Image.open(BytesIO(image_bytes))
        except Exception:
            return None

    model = genai.GenerativeModel("gemini-1.5-flash")

    cls_prompt = f"""
Nhìn ảnh sản phẩm giày. Hãy chọn 1 nhãn DUY NHẤT trong danh sách:
{s.strip() for s in SHOE_TYPES}

Quy tắc:
- Trả về đúng 1 từ khóa duy nhất (không giải thích).
- Nếu là giày tây/loafer/oxford/derby -> "leather"
- Nếu là sneaker thường -> "sneaker"
- Nếu là giày chạy -> "runner"
- Nếu là dép/sandal -> "sandals"
- Nếu là boot -> "boots"
- Nếu vibe sang trọng tối giản (giày tây cao cấp) -> "luxury"
- Nếu kiểu casual everyday không rõ -> "casual"
""".strip()

    try:
        resp = model.generate_content([cls_prompt, img])
        out = (resp.text or "").strip().lower()
        out = re.sub(r"[^a-z]", "", out)
        if out in SHOE_TYPES:
            return out
        # map nhẹ
        if out == "boot":
            return "boots"
        return None
    except Exception:
        return None


# =========================
# BUILD PROMPTS
# =========================
def build_prompt_p1(shoe_type, tone, shoe_name, dialogue_text):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
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

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- KEEP 100% shoe identity (shape, sole, panels, stitching, proportions).
- NO redesign • NO deformation • NO guessing • NO color shift
- If shoe has laces → keep laces in ALL frames; if NO laces → ABSOLUTELY NO laces.

PRODUCT (INTERNAL)
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

def build_prompt_p2(shoe_type, tone, shoe_name, dialogue_text):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Thông tin trong video mang tính tham khảo."

    return f"""
SORA VIDEO PROMPT — PROMPT 2 (CÓ CAMEO) — TIMELINE LOCK 10s
CAMEO VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- KEEP 100% shoe identity (shape, sole, panels, stitching, proportions).
- NO redesign • NO deformation • NO guessing • NO color shift
- If shoe has laces → keep laces in ALL frames; if NO laces → ABSOLUTELY NO laces.

PRODUCT (INTERNAL)
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

SCENE
- {scene_line(s)}

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
# UI
# =========================
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("📤 Tải ảnh giày", type=["jpg", "png", "jpeg"])
    mode = st.radio("Chọn loại prompt", ["PROMPT 1 – Không cameo", "PROMPT 2 – Có cameo"], index=1)
    tone = st.selectbox("Chọn tone thoại", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    count = st.slider("Số lượng prompt", 1, 10, 5)

with right:
    st.subheader("⚡ AI MODE (Gemini Free)")
    ai_mode = st.checkbox("Bật AI MODE (tự viết thoại + đoán shoe_type theo ẢNH)", value=False)
    api_key = st.text_input("Gemini API Key (dán vào đây)", type="password", help="Không cần nếu tắt AI MODE.")

    if ai_mode:
        if not GEMINI_OK:
            st.error("❌ Chưa cài google-generativeai. Xem requirements.txt bên dưới.")
        elif not PIL_OK:
            st.error("❌ Chưa cài Pillow. Xem requirements.txt bên dưới.")
        elif not api_key:
            st.warning("⚠️ AI MODE đang bật nhưng chưa có API key → sẽ fallback CSV.")
        else:
            st.success("✅ AI MODE sẵn sàng (có key).")

    st.divider()
    st.subheader("📌 Hướng dẫn nhanh")
    st.write("1) Upload ảnh • 2) Chọn Prompt 1/2 • 3) Chọn tone • 4) Bấm SINH • 5) Bấm tab 1..N để xem & COPY")
    st.caption(f"Dialogues columns: {dialogue_cols}")
    st.caption(f"Scenes columns: {scene_cols}")

st.divider()


if uploaded:
    shoe_name = normalize_filename_to_shoename(uploaded.name)

    # đọc bytes
    image_bytes = uploaded.getvalue()

    # shoe_type: AI đoán từ ảnh (nếu bật + có key), còn lại auto theo tên file / manual
    # manual chọn tay luôn cho chắc
    shoe_type_choice = st.selectbox("Chọn shoe_type (Auto hoặc chọn tay)", ["Auto"] + SHOE_TYPES, index=0)

    detected_by_ai = None
    if ai_mode and api_key and GEMINI_OK and PIL_OK:
        # chỉ đoán 1 lần cho mỗi upload session
        cache_k = f"ai_detect_{shoe_name}_{len(image_bytes)}"
        if cache_k not in st.session_state:
            detected_by_ai = gemini_detect_shoe_type_from_image(api_key, image_bytes)
            st.session_state[cache_k] = detected_by_ai
        else:
            detected_by_ai = st.session_state[cache_k]

    # fallback cũ: dựa tên file (nhưng chỉ dùng khi không có AI)
    def detect_shoe_from_filename(name):
        n = (name or "").lower()
        if "loafer" in n or "loafers" in n or "horsebit" in n or "oxford" in n or "derby" in n:
            return "leather"
        if "sandal" in n or "dep" in n:
            return "sandals"
        if "boot" in n:
            return "boots"
        if "run" in n or "runner" in n:
            return "runner"
        if "lux" in n:
            return "luxury"
        if "casual" in n:
            return "casual"
        return "sneaker"

    guessed_from_name = detect_shoe_from_filename(uploaded.name)

    if shoe_type_choice == "Auto":
        if detected_by_ai in SHOE_TYPES:
            shoe_type = detected_by_ai
            st.success(f"👟 shoe_type: **{shoe_type}** (AI đoán từ ẢNH ✅)")
        else:
            shoe_type = guessed_from_name
            st.info(f"👟 shoe_type: **{shoe_type}** (Auto theo tên file)")
    else:
        shoe_type = shoe_type_choice
        st.success(f"👟 shoe_type: **{shoe_type}** (chọn tay)")

    st.caption(f"🧾 shoe_name (từ tên file): {shoe_name}")

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            # lấy scene trước để làm hint cho AI thoại
            s_pool = filter_scenes_by_shoe_type(shoe_type)
            s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
            s_hint = scene_line(s)

            # thoại: AI nếu bật + có key, else CSV 3 dòng
            d_pool = filter_dialogues(shoe_type, tone)
            dialogue_text = None

            if ai_mode and api_key and GEMINI_OK:
                dialogue_text = gemini_generate_3_sentences(
                    api_key=api_key,
                    shoe_type=shoe_type,
                    tone=tone,
                    scene_hint=s_hint,
                    shoe_name=shoe_name
                )

            if not dialogue_text:
                dialogue_text = get_3_lines_from_csv(d_pool, tone)

            # build prompt dùng lại scene s vừa chọn (để match)
            if mode.startswith("PROMPT 1"):
                # build prompt 1 nhưng ép scene s vừa pick
                disclaimer = random.choice(disclaimers_p1 if disclaimers_p1 else DISCLAIMER_P1_FALLBACK)
                p = f"""
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
- KEEP 100% shoe identity (shape, sole, panels, stitching, proportions).
- NO redesign • NO deformation • NO guessing • NO color shift
- If shoe has laces → keep laces in ALL frames; if NO laces → ABSOLUTELY NO laces.

PRODUCT (INTERNAL)
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

SCENE
- {s_hint}

AUDIO TIMELINE
0.0–1.2s: Không thoại, ambient + nhạc nền rất nhẹ
1.2–6.9s: VOICE ON (3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.2–6.9s]
{dialogue_text}

SAFETY / MIỄN TRỪ
- {disclaimer}
""".strip()
            else:
                disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Thông tin trong video mang tính tham khảo."
                p = f"""
SORA VIDEO PROMPT — PROMPT 2 (CÓ CAMEO) — TIMELINE LOCK 10s
CAMEO VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- KEEP 100% shoe identity (shape, sole, panels, stitching, proportions).
- NO redesign • NO deformation • NO guessing • NO color shift
- If shoe has laces → keep laces in ALL frames; if NO laces → ABSOLUTELY NO laces.

PRODUCT (INTERNAL)
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

SCENE
- {s_hint}

AUDIO TIMELINE
0.0–1.0s: Không thoại, ambient + nhạc nền rất nhẹ
1.0–6.9s: VOICE ON (3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.0–6.9s]
{dialogue_text}

SAFETY / MIỄN TRỪ (PROMPT 2)
- {disclaimer}
""".strip()

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
    # xoá cache ai detect nhẹ
    for k in list(st.session_state.keys()):
        if str(k).startswith("ai_detect_"):
            del st.session_state[k]
    st.success("✅ Đã reset")


# =========================
# REQUIREMENTS HINT
# =========================
with st.expander("📦 requirements.txt (nếu bật AI MODE mà báo thiếu thư viện)"):
    st.code(
        "\n".join([
            "streamlit",
            "pandas",
            "pillow",
            "google-generativeai"
        ]),
        language="text"
    )
