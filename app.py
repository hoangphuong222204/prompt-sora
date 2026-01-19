import streamlit as st
import pandas as pd
import random
import base64
from pathlib import Path
import re
from io import BytesIO

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
    """
    Hỗ trợ mọi kiểu header cho disclaimer_prompt2.csv
    - ưu tiên cột 'disclaimer'
    - nếu không có -> thử text/content/note...
    - nếu vẫn không -> nếu cột 1 là id -> lấy cột 2, else lấy cột cuối
    """
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
# MEMORY – CHỐNG TRÙNG + PROMPTS
# =========================
if "used_dialogue_ids" not in st.session_state:
    st.session_state.used_dialogue_ids = set()
if "used_scene_ids" not in st.session_state:
    st.session_state.used_scene_ids = set()
if "generated_prompts" not in st.session_state:
    st.session_state.generated_prompts = []
if "used_voice_lines" not in st.session_state:
    st.session_state.used_voice_lines = set()

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
    n = (name or "").lower()
    n = re.sub(r"\.(jpg|jpeg|png|webp|bmp)$", "", n)
    n = re.sub(r"[^a-z0-9_ -]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def extract_shoe_name(name: str) -> str:
    # Lấy tên "đẹp" từ filename: bỏ đuôi, bỏ timestamp dài, bỏ cụm vô nghĩa
    n = normalize_filename(name)
    # bỏ chuỗi số dài (timestamp)
    n = re.sub(r"\b\d{8,}\b", "", n).strip()
    # rút gọn
    if not n:
        return "uploaded_shoe"
    # giới hạn độ dài
    return n[:60]

# =========================
# SMART AUTO DETECT shoe_type (FIXED)
# =========================
KEYWORD_RULES = {
    "leather": [
        "loafer", "loafers", "horsebit", "bit", "oxford", "derby", "monk", "monkstrap",
        "brogue", "formal", "dress", "moc", "moccasin", "mocassin", "giay-da", "giay da",
        "da-nam", "da nam", "cong so", "cong-so", "tay", "slipon", "slip-on"
    ],
    "luxury": [
        "lux", "luxury", "premium", "quiet", "boutique", "highend", "high-end", "handmade",
        "classic", "elegant", "formal-lux"
    ],
    "boots": [
        "boot", "boots", "chelsea", "combat", "ankleboot", "ankle-boot", "chukka"
    ],
    "sandals": [
        "sandal", "sandals", "dep", "dép", "slide", "slides", "slipper", "flipflop", "flip-flop"
    ],
    "runner": [
        "runner", "running", "run", "jog", "training", "sport", "the thao", "the-thao", "gym"
    ],
    "sneaker": [
        "sneaker", "sneakers", "tennis", "casual-sneaker", "street", "streetwear"
    ],
    "casual": [
        "casual", "daily", "everyday", "basic", "lifestyle"
    ],
}

def smart_detect_shoe_type(filename: str):
    """
    Trả về: (shoe_type, confidence(0-100), reason)
    """
    n = normalize_filename(filename)
    if not n:
        return "sneaker", 30, "Không có tên file để suy luận"

    scores = {k: 0 for k in SHOE_TYPES}
    hits = {k: [] for k in SHOE_TYPES}

    # ưu tiên mạnh cho leather/luxury khi có keyword rõ
    for stype, kws in KEYWORD_RULES.items():
        for kw in kws:
            # match theo word-boundary mềm (có thể có dấu gạch)
            if kw in n:
                w = 8
                if stype in ["leather", "luxury"]:
                    w = 12
                if stype in ["boots", "sandals"]:
                    w = 10
                scores[stype] += w
                hits[stype].append(kw)

    # Heuristic nâng cấp: nếu có "da" hoặc "cong so" -> leather
    if re.search(r"\bda\b", n) or "giay da" in n or "cong so" in n or "công sở" in n:
        scores["leather"] += 10
        hits["leather"].append("da/cong-so")

    # Nếu leather mạnh thì giảm khả năng sneaker/runner
    if scores["leather"] >= 12:
        scores["sneaker"] = max(0, scores["sneaker"] - 6)
        scores["runner"] = max(0, scores["runner"] - 6)

    # Quy tắc ưu tiên: nếu leather và luxury đều có điểm, ưu tiên luxury khi luxury >= leather
    # (vì nhiều file đặt tên premium/quiet luxury cho giày da)
    best = max(scores.items(), key=lambda x: x[1])[0]
    best_score = scores[best]

    # Nếu không có keyword gì -> mặc định sneaker nhưng confidence thấp
    if best_score <= 0:
        return "sneaker", 25, "Không có keyword nhận dạng (fallback sneaker)"

    # confidence
    # max theoretical ~ 40-60; ta clamp về 40..95
    conf = min(95, max(40, int(best_score * 4)))
    reason = f"Match: {', '.join(hits[best][:6])}" if hits[best] else "Heuristic score"
    return best, conf, reason

# =========================
# THOẠI: ép ra ĐÚNG 3 câu, không na ná nhau
# =========================
TONE_LINE_BANK = {
    "Tự tin": [
        "Mình thích cảm giác gọn gàng, bước đi nhìn cũng rõ ràng hơn.",
        "Form lên chân ổn, phối đồ cũng dễ mà không cần cầu kỳ.",
        "Đi cả ngày vẫn thấy nhịp chân khá thoải mái.",
        "Nhìn tổng thể sạch sẽ, hợp kiểu mặc đơn giản.",
        "Mình chọn đôi này khi muốn mọi thứ gọn và chắc.",
        "Cảm giác di chuyển mượt, không bị vướng nhịp.",
        "Đứng dáng lên nhìn tự tin hơn hẳn."
    ],
    "Truyền cảm": [
        "Có những đôi mang vào là mood tự nhiên dịu lại.",
        "Nhìn kỹ mới thấy cái hay nằm ở sự tinh giản.",
        "Mình thích cảm giác vừa vặn, nhẹ nhàng khi di chuyển.",
        "Không cần nổi bật quá, nhưng càng nhìn càng có gu.",
        "Ánh sáng lên form nhìn rất êm và mềm mắt.",
        "Đi chậm thôi mà thấy mọi thứ cân bằng hơn."
    ],
    "Mạnh mẽ": [
        "Mình cần sự chắc chân để giữ nhịp cả ngày.",
        "Bước nhanh hơn một chút vẫn thấy ổn định.",
        "Nhịp đi dứt khoát, cảm giác gọn và vững.",
        "Ngày bận rộn thì mình ưu tiên kiểu chắc chắn như vậy.",
        "Di chuyển liên tục mà vẫn giữ được phong thái.",
        "Cảm giác bám nhịp tốt, không bị chông chênh."
    ],
    "Lãng mạn": [
        "Chiều xuống là mình thích đi chậm để cảm nhận không khí.",
        "Nhịp bước thư thả làm mọi thứ nhẹ hơn.",
        "Có cảm giác tinh tế rất vừa đủ, không phô trương.",
        "Không gian yên yên là tự nhiên thấy dễ chịu.",
        "Mình thích kiểu đơn giản mà vẫn có cảm xúc.",
        "Đi vài bước thôi mà mood đã khác."
    ],
    "Tự nhiên": [
        "Mình ưu tiên sự thoải mái, mang là muốn đi tiếp.",
        "Cảm giác nhẹ nhàng, hợp những ngày muốn thả lỏng.",
        "Nhìn tổng thể tự nhiên, không bị gò bó.",
        "Đi lâu một chút vẫn thấy dễ chịu.",
        "Chuyển động nhẹ, nhịp chân êm và đều.",
        "Mình thích kiểu đơn giản, gần gũi."
    ],
}

def split_sentences(text: str):
    # tách câu theo . ! ? (giữ sạch)
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    # nếu người dùng viết không có dấu chấm -> coi như 1 câu
    return parts

def pick_unique_voice_line(pool, used_set):
    candidates = [x for x in pool if x not in used_set]
    if not candidates:
        used_set.clear()
        candidates = pool[:]
    line = random.choice(candidates)
    used_set.add(line)
    return line

def get_dialogue_text(row, tone):
    """
    Đảm bảo output: ĐÚNG 3 câu, không lặp ý kiểu đảo lại.
    - ưu tiên lấy từ CSV (cột dialogue/text/...)
    - nếu CSV chỉ có 1 câu -> bổ sung 2 câu từ bank theo tone (unique)
    - nếu CSV có 2 câu -> bổ sung 1 câu từ bank
    - nếu CSV có >=3 câu -> lấy 3 câu đầu tiên khác nhau (random)
    """
    csv_text = ""
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row:
            t = safe_text(row.get(col))
            if t:
                csv_text = t
                break

    bank = TONE_LINE_BANK.get(tone, TONE_LINE_BANK["Tự tin"])

    # nếu CSV có text
    if csv_text:
        sents = split_sentences(csv_text)
        # nếu CSV không có dấu câu (1 câu dài) -> coi là 1
        if len(sents) == 0:
            sents = [csv_text.strip()]

        # làm sạch trùng
        uniq = []
        for s in sents:
            ss = s.strip()
            if ss and ss not in uniq:
                uniq.append(ss)

        if len(uniq) >= 3:
            # chọn 3 câu khác nhau, random để không “na ná”
            chosen = random.sample(uniq, 3)
            return " ".join([c if c.endswith((".", "!", "?")) else c + "." for c in chosen])

        if len(uniq) == 2:
            extra = pick_unique_voice_line(bank, st.session_state.used_voice_lines)
            chosen = [uniq[0], uniq[1], extra]
            return " ".join([c if c.endswith((".", "!", "?")) else c + "." for c in chosen])

        if len(uniq) == 1:
            extra1 = pick_unique_voice_line(bank, st.session_state.used_voice_lines)
            extra2 = pick_unique_voice_line(bank, st.session_state.used_voice_lines)
            chosen = [uniq[0], extra1, extra2]
            return " ".join([c if c.endswith((".", "!", "?")) else c + "." for c in chosen])

    # fallback: không có csv_text
    extra1 = pick_unique_voice_line(bank, st.session_state.used_voice_lines)
    extra2 = pick_unique_voice_line(bank, st.session_state.used_voice_lines)
    extra3 = pick_unique_voice_line(bank, st.session_state.used_voice_lines)
    chosen = [extra1, extra2, extra3]
    return " ".join([c if c.endswith((".", "!", "?")) else c + "." for c in chosen])

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
# BUILD PROMPTS (FIX: include shoe_name + shoe_type)
# =========================
def build_prompt_p1(shoe_name, shoe_type, tone):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
    disclaimer = random.choice(disclaimers_p1 if disclaimers_p1 else DISCLAIMER_P1_FALLBACK)

    dialogue_text = get_dialogue_text(d, tone)

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

PRODUCT (for consistency, not for selling)
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

SCENE
- {scene_line(s)}

AUDIO TIMELINE
0.0–1.2s: Không thoại, ambient + nhạc nền rất nhẹ
1.2–6.9s: VOICE ON (ĐÚNG 3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.2–6.9s]
{dialogue_text}

SAFETY / MIỄN TRỪ
- {disclaimer}
""".strip()

def build_prompt_p2(shoe_name, shoe_type, tone):
    s_pool = filter_scenes_by_shoe_type(shoe_type)
    d_pool = filter_dialogues(shoe_type, tone)

    s = pick_unique(s_pool, st.session_state.used_scene_ids, "id")
    d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
    disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Thông tin chi tiết vui lòng xem trong giỏ hàng."

    dialogue_text = get_dialogue_text(d, tone)

    return f"""
SORA VIDEO PROMPT — PROMPT 2 (CÓ CAMEO) — TIMELINE LOCK 10s
CAMEO VOICE ID: {CAMEO_VOICE_ID}

VIDEO SETUP
- Video dọc 9:16 — 10s — Ultra Sharp 4K
- Video thật, chuyển động mượt (không ảnh tĩnh)
- NO text • NO logo • NO watermark
- NO blur • NO haze • NO glow

CAMEO SETUP (SAFE)
- Cameo xuất hiện tự nhiên, không CTA, không bán hàng
- Voice nói kiểu chia sẻ trải nghiệm đời thường

SHOE REFERENCE — ABSOLUTE LOCK
- Use ONLY the uploaded shoe image as reference.
- KEEP 100% shoe identity (shape, sole, panels, stitching, proportions).
- NO redesign • NO deformation • NO guessing • NO color shift

PRODUCT (for consistency, not for selling)
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

SCENE
- {scene_line(s)}

AUDIO TIMELINE
0.0–1.0s: Không thoại, ambient + nhạc nền rất nhẹ
1.0–6.9s: VOICE ON (ĐÚNG 3 câu, đời thường, chia sẻ trải nghiệm)
6.9–10.0s: VOICE OFF (im hẳn) + fade-out 9.2–10.0s

[VOICEOVER {CAMEO_VOICE_ID} | 1.0–6.9s]
{dialogue_text}

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
    st.write("1) Upload ảnh • 2) Chọn Prompt 1/2 • 3) Chọn tone • 4) Bấm SINH • 5) Bấm số 1..N để xem & COPY")
    st.caption(f"Dialogues columns: {dialogue_cols}")
    st.caption(f"Scenes columns: {scene_cols}")
    if Path("disclaimer_prompt1.csv").exists():
        st.success("✅ Đã có disclaimer_prompt1.csv (Prompt 1 sẽ random theo file).")
    else:
        st.info("ℹ️ Chưa có disclaimer_prompt1.csv (Prompt 1 dùng danh sách dự phòng).")

st.divider()

if uploaded:
    # shoe_name lấy từ filename (không phụ thuộc shoe_type)
    shoe_name = extract_shoe_name(uploaded.name)

    # Smart auto detect shoe_type
    auto_type, auto_conf, auto_reason = smart_detect_shoe_type(uploaded.name)

    st.info(f"🧾 **shoe_name (lấy từ tên file):** `{shoe_name}`")

    shoe_type_choice = st.selectbox(
        "Chọn shoe_type (Auto hoặc chọn tay)",
        ["Auto"] + SHOE_TYPES,
        index=0
    )
    shoe_type = auto_type if shoe_type_choice == "Auto" else shoe_type_choice

    if shoe_type_choice == "Auto":
        # Cảnh báo khi confidence thấp
        if auto_conf < 60:
            st.warning(
                f"⚠️ Auto đoán **{auto_type}** nhưng độ tin cậy thấp (**{auto_conf}%**). "
                f"Lý do: {auto_reason}. Khuyên chồng chọn tay cho chắc."
            )
        else:
            st.success(f"✅ Auto đoán shoe_type: **{auto_type}** ({auto_conf}%) • {auto_reason}")
    else:
        # Nếu user chọn tay khác auto thì báo
        if shoe_type_choice != auto_type and auto_conf >= 60:
            st.warning(f"ℹ️ Chồng chọn tay **{shoe_type_choice}** khác Auto (**{auto_type}**). OK, app sẽ dùng chọn tay.")
        st.success(f"👟 shoe_type (chọn tay): **{shoe_type_choice}**")

    btn_label = "🎬 SINH PROMPT 1" if mode.startswith("PROMPT 1") else "🎬 SINH PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        # reset used_voice_lines mỗi lần sinh batch để 1 batch không trùng câu quá nhiều
        st.session_state.used_voice_lines.clear()

        for _ in range(count):
            p = build_prompt_p1(shoe_name, shoe_type, tone) if mode.startswith("PROMPT 1") else build_prompt_p2(shoe_name, shoe_type, tone)
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
    st.session_state.used_voice_lines.clear()
    st.session_state.generated_prompts = []
    st.success("✅ Đã reset")
