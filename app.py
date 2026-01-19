import streamlit as st
import pandas as pd
import random
import base64
import re
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition (PRO)", layout="wide")
st.title("🎬 Sora Prompt Studio Pro – Director Edition (PRO)")
st.caption("Prompt 1 & 2 • Timeline thoại chuẩn • Không trùng • TikTok Shop SAFE • PRO: Auto-hide AI + Gemini debug logs")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]

BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path(".")
REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

# =========================
# PRO: detect library availability
# =========================
def gemini_available() -> bool:
    try:
        import google.generativeai  # noqa: F401
        return True
    except Exception:
        return False

GEMINI_READY = gemini_available()

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
missing = [f for f in REQUIRED_FILES if not (BASE_DIR / f).exists()]
if missing:
    st.error(f"❌ Thiếu file: {', '.join(missing)} (phải nằm cùng thư mục app.py)")
    st.stop()

# =========================
# LOAD CSV (robust)
# =========================
def _ensure_id(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c.strip() for c in df.columns.tolist()]
    df.columns = cols
    if "id" not in df.columns:
        df.insert(0, "id", [str(i + 1) for i in range(len(df))])
    df["id"] = df["id"].astype(str)
    return df

@st.cache_data
def load_dialogues():
    df = pd.read_csv(str(BASE_DIR / "dialogue_library.csv"))
    df = _ensure_id(df)
    return df.to_dict(orient="records"), [c.strip() for c in df.columns.tolist()]

@st.cache_data
def load_scenes():
    df = pd.read_csv(str(BASE_DIR / "scene_library.csv"))
    df = _ensure_id(df)
    return df.to_dict(orient="records"), [c.strip() for c in df.columns.tolist()]

@st.cache_data
def load_disclaimer_prompt2_flexible():
    df = pd.read_csv(str(BASE_DIR / "disclaimer_prompt2.csv"))
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

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2_flexible()

# =========================
# SESSION – ANTI DUP + DEBUG
# =========================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "gemini_debug" not in st.session_state:
    st.session_state.gemini_debug = None  # will store dict logs

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

def pick_unique(pool: List[dict], used_ids: set, key: str):
    if not pool:
        return {}
    if key not in pool[0]:
        return random.choice(pool)
    items = [x for x in pool if str(x.get(key, "")).strip() not in used_ids]
    if not items:
        used_ids.clear()
        items = pool[:]
    item = random.choice(items)
    used_ids.add(str(item.get(key, "")).strip())
    return item

def scene_line(scene: dict) -> str:
    parts = [
        safe_text(scene.get("lighting")),
        safe_text(scene.get("location")),
        safe_text(scene.get("motion")),
        safe_text(scene.get("weather")),
    ]
    mood = safe_text(scene.get("mood"))
    if mood:
        parts.append(f"mood {mood}")
    return " • ".join([p for p in parts if p]).strip(" •")

def filter_scenes_by_shoe_type(shoe_type: str):
    if scenes and "shoe_type" in scenes[0]:
        f = [s for s in scenes if safe_text(s.get("shoe_type")).lower() == shoe_type.lower()]
        return f if f else scenes
    return scenes

def filter_dialogues(shoe_type: str, tone: str):
    pool = dialogues
    if dialogues and "tone" in dialogues[0]:
        tone_f = [d for d in dialogues if safe_text(d.get("tone")) == tone]
        pool = tone_f if tone_f else dialogues
    if pool and "shoe_type" in pool[0]:
        shoe_f = [d for d in pool if safe_text(d.get("shoe_type")).lower() == shoe_type.lower()]
        pool = shoe_f if shoe_f else pool
    return pool

# =========================
# FILENAME DETECT (fallback)
# =========================
def detect_shoe_from_filename(name: str) -> str:
    n = (name or "").lower()
    rules = [
        ("boots",  ["boot", "chelsea", "combat", "martin"]),
        ("sandals",["sandal", "sandals", "dep", "dép", "slipper", "slides"]),
        ("leather",["loafer", "loafers", "moc", "moccasin", "horsebit", "oxford", "derby", "tassel", "brogue",
                    "giaytay", "giày tây", "giay_da", "giayda", "giay da"]),
        ("runner", ["runner", "running", "jog", "marathon", "gym", "train", "thethao", "thể thao", "sport"]),
        ("casual", ["casual", "daily", "everyday", "basic"]),
        ("luxury", ["lux", "premium", "quietlux", "quiet_lux", "highend", "boutique"]),
        ("sneaker",["sneaker", "sneakers", "kicks", "street"]),
    ]
    for shoe_type, keys in rules:
        if any(k in n for k in keys):
            return shoe_type
    return "sneaker"

# =========================
# GEMINI VISION DETECT (PRO + DEBUG LOGS)
# =========================
def gemini_detect_shoe_type(img: Image.Image, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Returns:
      {"shoe_type": str, "confidence": float, "raw": str}
    On failure -> returns None (and writes debug to st.session_state.gemini_debug)
    """
    st.session_state.gemini_debug = None

    api_key = (api_key or "").strip()
    if not api_key:
        st.session_state.gemini_debug = {"ok": False, "stage": "no_key", "error": "No API key provided."}
        return None

    if not GEMINI_READY:
        st.session_state.gemini_debug = {
            "ok": False,
            "stage": "missing_library",
            "error": "google-generativeai is not installed on this server.",
            "hint": "Add google-generativeai to requirements.txt then redeploy/restart."
        }
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        # NOTE: using flash for speed/cost
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
Bạn là hệ thống phân loại shoe_type cho TikTok prompts.

Hãy nhìn ảnh giày và trả về JSON DUY NHẤT theo format:
{{
  "shoe_type": "sneaker|runner|leather|casual|sandals|boots|luxury|unknown",
  "confidence": 0.0-1.0
}}

Quy tắc:
- "leather" cho loafer/oxford/derby/dress shoes.
- "runner" cho running/training.
- "sneaker" cho street sneaker.
- "luxury" nếu rõ phong cách high-end.
- Nếu không chắc: "unknown" và confidence thấp.
Chỉ trả JSON, không thêm chữ khác.
""".strip()

        resp = model.generate_content([prompt, img])
        raw = (getattr(resp, "text", "") or "").strip()

        st.session_state.gemini_debug = {"ok": True, "stage": "response_received", "raw_text": raw[:3000]}

        # parse JSON
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            st.session_state.gemini_debug = {
                "ok": False,
                "stage": "parse_json",
                "error": "No JSON object found in model response.",
                "raw_text": raw[:3000]
            }
            return {"shoe_type": "unknown", "confidence": 0.0, "raw": raw}

        obj = json.loads(m.group(0))
        shoe_type = str(obj.get("shoe_type", "unknown")).strip().lower()
        conf = float(obj.get("confidence", 0.0) or 0.0)

        if shoe_type not in SHOE_TYPES and shoe_type != "unknown":
            shoe_type = "unknown"

        conf = max(0.0, min(1.0, conf))

        st.session_state.gemini_debug = {
            "ok": True,
            "stage": "parsed",
            "shoe_type": shoe_type,
            "confidence": conf,
            "raw_text": raw[:3000]
        }

        return {"shoe_type": shoe_type, "confidence": conf, "raw": raw}

    except Exception as e:
        st.session_state.gemini_debug = {
            "ok": False,
            "stage": "exception",
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc()[:4000]
        }
        return None

def hybrid_pick(ai_result: Optional[dict], fallback_type: str) -> str:
    if not ai_result or not isinstance(ai_result, dict):
        return fallback_type
    ai_type = str(ai_result.get("shoe_type", "unknown")).strip().lower()
    conf = float(ai_result.get("confidence", 0.0) or 0.0)
    if ai_type in SHOE_TYPES and conf >= 0.60:
        return ai_type
    return fallback_type

# =========================
# DIALOGUE BANK
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
                    return f"{base}. {mid} {close}"

    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])
    a = random.choice(bank["open"])
    b = random.choice(bank["mid"])
    c = random.choice(bank["close"])
    while b == a and len(bank["mid"]) > 1:
        b = random.choice(bank["mid"])
    while c in (a, b) and len(bank["close"]) > 1:
        c = random.choice(bank["close"])
    return f"{a} {b} {c}"

# =========================
# PROMPTS
# =========================
def build_prompt_p1(shoe_type: str, tone: str, scene: dict, dialogue_text: str, shoe_name: str) -> str:
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
# SIDEBAR: GEMINI KEY + DEBUG TOGGLE
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

    st.markdown("---")
    show_debug = st.checkbox("🧪 Hiện debug Gemini (PRO)", value=False)
    if not GEMINI_READY:
        st.warning("⚠️ Server thiếu thư viện google-generativeai → chế độ AI sẽ tự ẩn.\n\nThêm vào requirements.txt:\n- google-generativeai")

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
    st.info("ℹ️ Prompt 1: KHÔNG cần miễn trừ. Prompt 2: có miễn trừ từ disclaimer_prompt2.csv")

st.divider()

if uploaded:
    shoe_name = Path(uploaded.name).stem.replace("_", " ").strip()
    img = Image.open(uploaded).convert("RGB")

    detected_filename = detect_shoe_from_filename(uploaded.name)

    # ===== Gemini detect (only if lib ready + key exists) =====
    detected_ai = None
    ai_error = ""

    if GEMINI_READY and st.session_state.gemini_api_key.strip():
        detected_ai = gemini_detect_shoe_type(img, st.session_state.gemini_api_key)
        if detected_ai is None:
            ai_error = "Gemini detect lỗi → fallback Auto theo tên file."
    elif not GEMINI_READY:
        ai_error = "Server thiếu thư viện google-generativeai → AI bị ẩn, fallback Auto."
    else:
        ai_error = "Chưa có Gemini key → AI không chạy, fallback Auto theo tên file."

    # ===== PRO: auto-hide AI mode if missing library =====
    modes = ["Auto", "Chọn tay"]
    if GEMINI_READY:
        modes.insert(0, "AI")

    default_mode = "AI" if (GEMINI_READY and detected_ai) else "Auto"

    shoe_mode = st.selectbox(
        "Chọn chế độ shoe_type",
        modes,
        index=modes.index(default_mode),
        help="AI: Gemini Vision | Auto: đoán theo tên file | Chọn tay: bạn tự chọn"
    )

    if shoe_mode == "Chọn tay":
        default_idx = SHOE_TYPES.index("leather") if "leather" in SHOE_TYPES else 0
        shoe_type = st.selectbox("Chọn shoe_type (tay)", SHOE_TYPES, index=default_idx)
        st.success(f"👟 Chọn tay: **{shoe_type}**")

    elif shoe_mode == "AI":
        if detected_ai and isinstance(detected_ai, dict):
            shoe_type = hybrid_pick(detected_ai, detected_filename)
            conf = float(detected_ai.get("confidence", 0.0) or 0.0)
            st.success(f"👟 AI detect shoe_type: **{shoe_type}** (conf: {conf:.2f})")
            if shoe_type == detected_filename and (detected_ai.get("shoe_type") in ["unknown", None] or conf < 0.60):
                st.warning("AI chưa chắc → dùng fallback theo tên file.")
        else:
            shoe_type = detected_filename
            st.warning(ai_error)
            st.info(f"Fallback Auto (tên file): **{detected_filename}**")

    else:  # Auto
        shoe_type = detected_filename
        st.info(f"👟 Auto theo tên file: **{shoe_type}**")

    st.caption(f"shoe_name (tên file): {shoe_name}")

    # ===== PRO: show debug logs =====
    if show_debug:
        dbg = st.session_state.get("gemini_debug")
        with st.expander("🧪 Gemini Debug Logs (PRO)", expanded=True):
            st.write("GEMINI_READY:", GEMINI_READY)
            st.write("Has key:", bool(st.session_state.gemini_api_key.strip()))
            if dbg is None:
                st.info("Chưa có log (chưa gọi Gemini hoặc Gemini bị ẩn).")
            else:
                st.json(dbg)

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
    st.session_state.gemini_debug = None
    st.success("✅ Đã reset")
