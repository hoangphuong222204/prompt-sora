import streamlit as st
import pandas as pd
import random
import base64
import re
import json
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro - Director Edition", layout="wide")
st.title("Sora Prompt Studio Pro - Director Edition")
st.caption("Prompt 1 and 2 - Total 10s - Multi-scene - Voice timeline - Anti-duplicate - TikTok Shop SAFE")

CAMEO_VOICE_ID = "@phuongnghi18091991"
SHOE_TYPES = ["sneaker", "runner", "leather", "casual", "sandals", "boots", "luxury"]
REQUIRED_FILES = ["dialogue_library.csv", "scene_library.csv", "disclaimer_prompt2.csv"]

# =========================
# COPY BUTTON
# =========================
def copy_button(text: str, key: str):
    b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    html = f"""
    <button id="{key}" style="padding:8px 14px;border-radius:10px;border:1px solid #ccc;cursor:pointer;background:#fff;font-weight:700;">
      COPY
    </button>
    <span id="{key}_s" style="margin-left:8px;font-size:12px;"></span>
    <script>
    const btn = document.getElementById("{key}");
    const s = document.getElementById("{key}_s");
    btn.onclick = async () => {{
      try {{
        await navigator.clipboard.writeText(atob("{b64}"));
        s.innerText = "Copied";
        setTimeout(()=>s.innerText="",1500);
      }} catch(e) {{
        s.innerText = "Copy blocked";
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
    st.error("Missing files: " + ", ".join(missing) + " (must be in the same folder as app.py)")
    st.stop()

# =========================
# TEXT SAFETY + ENCODING REPAIR
# =========================
def _to_str(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v)
    if s.lower().strip() == "nan":
        return ""
    return s

def _repair_mojibake_once(s: str) -> str:
    if not s:
        return s
    # Common mojibake markers
    if ("Ã" in s) or ("Â" in s) or ("Ä" in s) or ("ðŸ" in s):
        # utf8 bytes decoded as latin1/cp1252
        for enc in ("latin1", "cp1252"):
            try:
                s2 = s.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
                if s2 and (s2.count("Ã") < s.count("Ã")):
                    return s2
            except Exception:
                pass
    return s

def safe_text(v) -> str:
    s = _to_str(v).strip()
    if not s:
        return ""
    # normalize
    try:
        s = unicodedata.normalize("NFC", s)
    except Exception:
        pass
    # repair mojibake multiple passes
    s1 = _repair_mojibake_once(s)
    s2 = _repair_mojibake_once(s1)
    s3 = _repair_mojibake_once(s2)
    s = (s3 or "").strip()
    try:
        s = unicodedata.normalize("NFC", s)
    except Exception:
        pass
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def dot(x: str) -> str:
    x = safe_text(x)
    if not x:
        return ""
    return x if x.endswith((".", "!", "?")) else x + "."

# =========================
# CSV READ BEST EFFORT
# =========================
def read_csv_best_effort(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    # final attempt
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Cannot read {path}. Last error: {last_err} / {e}")

@st.cache_data
def load_dialogues():
    df = read_csv_best_effort("dialogue_library.csv")
    cols = [c.strip() for c in df.columns.tolist()]
    rows = df.to_dict(orient="records")
    return rows, cols

@st.cache_data
def load_scenes():
    df = read_csv_best_effort("scene_library.csv")
    cols = [c.strip() for c in df.columns.tolist()]
    rows = df.to_dict(orient="records")
    return rows, cols

@st.cache_data
def load_disclaimer_prompt2_flexible():
    df = read_csv_best_effort("disclaimer_prompt2.csv")
    cols = [c.strip() for c in df.columns.tolist()]

    def clean_list(arr):
        out = []
        for x in arr:
            s = safe_text(x)
            if s:
                out.append(s)
        return out

    if "disclaimer" in cols:
        return clean_list(df["disclaimer"].dropna().astype(str).tolist())

    preferred = ["text", "mien_tru", "mien_tru_ngan", "note", "content", "noi_dung", "line"]
    for c in preferred:
        if c in cols:
            return clean_list(df[c].dropna().astype(str).tolist())

    if len(cols) >= 2 and cols[0].lower() in ["id", "stt", "no"]:
        return clean_list(df[cols[1]].dropna().astype(str).tolist())

    last = cols[-1]
    return clean_list(df[last].dropna().astype(str).tolist())

dialogues, dialogue_cols = load_dialogues()
scenes, scene_cols = load_scenes()
disclaimers_p2 = load_disclaimer_prompt2_flexible()

# =========================
# SESSION - ANTI DUP
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
# LOCALSTORAGE KEY (PERSIST IN BROWSER)
# =========================
def localstorage_get_key(ls_key: str = "GEMINI_API_KEY") -> str:
    html = f"""
    <script>
    const v = window.localStorage.getItem("{ls_key}") || "";
    const msg = {{type: "LS_GET", key: "{ls_key}", value: v}};
    window.parent.postMessage(msg, "*");
    </script>
    """
    st.components.v1.html(html, height=0)

def localstorage_set_key(value: str, ls_key: str = "GEMINI_API_KEY"):
    v = value.replace("\\", "\\\\").replace('"', '\\"')
    html = f"""
    <script>
    window.localStorage.setItem("{ls_key}", "{v}");
    const msg = {{type: "LS_SET", key: "{ls_key}", ok: true}};
    window.parent.postMessage(msg, "*");
    </script>
    """
    st.components.v1.html(html, height=0)

def localstorage_clear_key(ls_key: str = "GEMINI_API_KEY"):
    html = f"""
    <script>
    window.localStorage.removeItem("{ls_key}");
    const msg = {{type: "LS_CLEAR", key: "{ls_key}", ok: true}};
    window.parent.postMessage(msg, "*");
    </script>
    """
    st.components.v1.html(html, height=0)

# Try load from localStorage once
if "ls_loaded" not in st.session_state:
    st.session_state.ls_loaded = True
    localstorage_get_key("GEMINI_API_KEY")

# =========================
# UTILS
# =========================
def pick_unique(pool, used_ids: set, key: str):
    def get_id(x):
        val = safe_text(x.get(key))
        if val:
            return val
        return str(hash(str(x)))

    items = [x for x in pool if get_id(x) not in used_ids]
    if not items:
        used_ids.clear()
        items = pool[:]
    item = random.choice(items)
    used_ids.add(get_id(item))
    return item

def filter_scenes_by_shoe_type(shoe_type: str):
    f = [s for s in scenes if safe_text(s.get("shoe_type")).lower() == shoe_type.lower()]
    return f if f else scenes

def filter_dialogues(shoe_type: str, tone: str):
    tone_f = [d for d in dialogues if safe_text(d.get("tone")) == tone]
    if not tone_f:
        tone_f = dialogues
    shoe_f = [d for d in tone_f if safe_text(d.get("shoe_type")).lower() == shoe_type.lower()]
    return shoe_f if shoe_f else tone_f

def split_10s_timeline(n: int) -> List[Tuple[float, float]]:
    n = max(2, min(4, int(n)))
    if n == 2:
        cuts = [0.0, 5.0, 10.0]
    elif n == 3:
        cuts = [0.0, 3.3, 6.7, 10.0]
    else:
        cuts = [0.0, 2.5, 5.0, 7.5, 10.0]
    return [(cuts[i], cuts[i + 1]) for i in range(n)]

def pick_n_unique_scenes(shoe_type: str, n: int) -> List[dict]:
    pool = filter_scenes_by_shoe_type(shoe_type)
    out = []
    for _ in range(n):
        s = pick_unique(pool, st.session_state.used_scene_ids, "id")
        out.append(s)
    return out

# =========================
# FILENAME HEURISTIC
# =========================
def detect_shoe_from_filename(name: str) -> str:
    n = (name or "").lower()
    rules = [
        ("boots", ["boot", "chelsea", "combat", "martin"]),
        ("sandals", ["sandal", "sandals", "dep", "dep_", "slipper", "slides"]),
        ("leather", ["loafer", "loafers", "moc", "moccasin", "horsebit", "oxford", "derby", "tassel", "brogue",
                     "giaytay", "giay_tay", "giay_da", "giayda", "dressshoe"]),
        ("runner", ["runner", "running", "jog", "marathon", "gym", "train", "thethao", "sport"]),
        ("casual", ["casual", "daily", "everyday", "basic"]),
        ("luxury", ["lux", "premium", "quietlux", "quiet_lux", "highend", "boutique"]),
        ("sneaker", ["sneaker", "sneakers", "kicks", "street"])
    ]
    for shoe_type, keys in rules:
        if any(k in n for k in keys):
            return shoe_type
    if re.fullmatch(r"[\d_\-\.]+", n):
        return "leather"
    return "sneaker"

# =========================
# GEMINI HELPERS
# =========================
def _pick_gemini_model(genai) -> Optional[str]:
    try:
        models = genai.list_models()
        available = [m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
        if not available:
            return None
        preferred = [
            "models/gemini-2.5-flash",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-pro-vision",
        ]
        for p in preferred:
            if p in available:
                return p
        return available[0]
    except Exception:
        return None

def gemini_detect_shoe_type(img: Image.Image, api_key: str) -> Tuple[Optional[str], str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "NO_KEY"

    try:
        import google.generativeai as genai
    except Exception as e:
        return None, f"IMPORT_FAIL: {type(e).__name__}"

    try:
        genai.configure(api_key=api_key)
        picked = _pick_gemini_model(genai)
        if not picked:
            return None, "NO_MODELS"

        model = genai.GenerativeModel(picked)
        prompt = (
            "You are a shoe classification system.\n"
            "Return ONLY ONE label from this list:\n"
            + ", ".join(SHOE_TYPES) +
            "\nRules:\n"
            "- Return exactly one word.\n"
            "- No explanations.\n"
            "- Dress shoe, loafer, oxford, derby -> leather.\n"
            "- Sports shoe or sneaker -> sneaker or runner.\n"
        )

        resp = model.generate_content([prompt, img])
        text = (getattr(resp, "text", "") or "").strip().lower()
        raw = f"{picked} | {text}" if text else f"{picked} | EMPTY_TEXT"

        norm = re.sub(r"[^a-z_]", "", text)
        if norm in SHOE_TYPES:
            return norm, raw
        for t in SHOE_TYPES:
            if t in norm:
                return t, raw
        return None, raw

    except Exception as e:
        # quota, 404, etc.
        return None, f"CALL_FAIL: {type(e).__name__}: {e}"

def ai_generate_dialogue_lines(api_key: str, shoe_type: str, tone: str, n_lines: int) -> Optional[List[str]]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai
    except Exception:
        return None

    try:
        genai.configure(api_key=api_key)
        picked = _pick_gemini_model(genai)
        if not picked:
            return None

        model = genai.GenerativeModel(picked)

        want = '{"lines":["...","..."]}' if n_lines == 2 else '{"lines":["...","...","..."]}'
        prompt = (
            "Return JSON only: " + want + "\n"
            "Language: Vietnamese.\n"
            "Style: natural spoken, short experience sharing.\n"
            f"Tone: {tone}. Shoe type: {shoe_type}.\n"
            "Rules:\n"
            "- No price.\n"
            "- No discount.\n"
            "- No guarantee.\n"
            "- No strong call to action.\n"
            "- No brand names.\n"
            "- No sensitive material words.\n"
            "- Each line 6 to 12 words.\n"
            "- Lines must not repeat each other.\n"
        )

        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()

        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
        lines = data.get("lines")
        if not isinstance(lines, list):
            return None

        out = []
        for x in lines:
            s = dot(x)
            if s:
                out.append(s)

        if len(out) != n_lines:
            return None
        return out
    except Exception:
        return None

# =========================
# DIALOGUE BANK FALLBACK
# =========================
TONE_BANK = {
    "Tự tin": {
        "open": [
            "Hom nay minh chon kieu gon gang de tu tin hon.",
            "Minh thich cam giac buoc di gon va co nhip.",
            "Minh uu tien tong the sach, de phoi, nhin sang dang."
        ],
        "mid": [
            "Di mot luc thay nhip buoc deu, cam giac on dinh.",
            "Form len chan nhin gon, de di ca ngay.",
            "Cam giac di chuyen nhe, nhin rat gon gang."
        ],
        "close": [
            "Tong the toi gian nhung nhin tinh te.",
            "Minh thich kieu toi gian de tao phong cach.",
            "Voi minh, gon gang la du dep roi."
        ],
    },
    "Truyền cảm": {
        "open": [
            "Co nhung doi mang vao la thay de chiu hon.",
            "Minh thich cam giac nhe nhang, cham rai ma chi chu.",
            "Nhin ky moi thay cai hay nam o su tinh gian."
        ],
        "mid": [
            "Di cham thoi nhung cam giac rat thu thai.",
            "Minh thich nhip buoc em, tao cam giac de chiu.",
            "Cang nhin cang thay tong the hai hoa."
        ],
        "close": [
            "Moi buoc nhu giu lai chut binh yen.",
            "Vua du tinh te de nhin lau khong chan.",
            "Doi khi chi can vay la dep."
        ],
    },
    "Mạnh mẽ": {
        "open": [
            "Hom nay minh muon nhip buoc dut khoat hon.",
            "Minh thich cam giac chac chan khi di chuyen nhanh.",
            "Ngay ban ron thi minh can su gon va on dinh."
        ],
        "mid": [
            "Di nhanh van thay kiem soat tot, khong chong chenh.",
            "Nhip buoc chac, cam giac bam chan on.",
            "Gon gang giup minh tu tin hon khi di chuyen."
        ],
        "close": [
            "Tong the nhin khoe ma van sach.",
            "Gon, chac, de phoi, dung gu minh.",
            "Chi can on dinh la minh yen tam."
        ],
    },
    "Lãng mạn": {
        "open": [
            "Chieu nay ra ngoai chut, tu nhien mood nhe hon.",
            "Minh thich kieu di cham, nhin moi thu mem lai.",
            "Nhung ngay vay, minh uu tien cam giac thu thai."
        ],
        "mid": [
            "Nhip buoc nhe, nhin tong the rat hai hoa.",
            "Cam giac vua van khien minh muon di them chut nua.",
            "Don gian thoi nhung len hinh lai thay diu."
        ],
        "close": [
            "Cang toi gian cang de tao cam xuc rieng.",
            "Minh thich su tinh te nam o nhung thu gian don.",
            "Mot chut nhe nhang la du."
        ],
    },
    "Tự nhiên": {
        "open": [
            "Minh uu tien thoai mai, mang la muon di tiep.",
            "Hom nay minh chon phong cach tu nhien, khong cau ky.",
            "Di ra ngoai ma van thay nhe la minh thich."
        ],
        "mid": [
            "Cam giac di chuyen mem, de chiu.",
            "Nhin tong the rat tu nhien, khong bi gong.",
            "Minh thay hop nhung ngay muon tha long."
        ],
        "close": [
            "Gon gang vay thoi nhung de dung hang ngay.",
            "Minh thich kieu don gian ma nhin sach.",
            "Nhe nhang la du dep roi."
        ],
    }
}

def split_sentences(text: str) -> List[str]:
    t = safe_text(text)
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"[.!?]+", t) if p.strip()]
    return parts

def get_dialogue_from_csv(row: dict) -> str:
    for col in ["dialogue", "text", "line", "content", "script", "noi_dung"]:
        if col in row and safe_text(row.get(col)):
            return safe_text(row.get(col))
    return ""

def get_dialogue_3_sentences(row: dict, tone: str) -> str:
    candidate = get_dialogue_from_csv(row)
    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])

    if candidate:
        parts = split_sentences(candidate)
        if len(parts) >= 3:
            a, b, c = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            a, b = parts[0], parts[1]
            c = random.choice(bank["close"])
        elif len(parts) == 1:
            a = parts[0]
            b = random.choice(bank["mid"])
            c = random.choice(bank["close"])
        else:
            a = random.choice(bank["open"])
            b = random.choice(bank["mid"])
            c = random.choice(bank["close"])
    else:
        a = random.choice(bank["open"])
        b = random.choice(bank["mid"])
        c = random.choice(bank["close"])

    # anti duplicate simple
    if b.strip().lower() == a.strip().lower():
        b = random.choice(bank["mid"])
    if c.strip().lower() in {a.strip().lower(), b.strip().lower()}:
        c = random.choice(bank["close"])

    return f"{dot(a)}\n{dot(b)}\n{dot(c)}"

def get_dialogue_2_sentences(row: dict, tone: str) -> str:
    candidate = get_dialogue_from_csv(row)
    bank = TONE_BANK.get(tone, TONE_BANK["Tự tin"])

    if candidate:
        parts = split_sentences(candidate)
        if len(parts) >= 2:
            a, b = parts[0], parts[1]
        elif len(parts) == 1:
            a = parts[0]
            b = random.choice(bank["mid"])
        else:
            a = random.choice(bank["open"])
            b = random.choice(bank["mid"])
    else:
        a = random.choice(bank["open"])
        b = random.choice(bank["mid"])

    if b.strip().lower() == a.strip().lower():
        b = random.choice(bank["mid"])

    return f"{dot(a)}\n{dot(b)}"

# =========================
# PROMPT BUILDERS (NO MARKDOWN TABLE, NO EMOJI, NO SPECIAL ARROWS)
# =========================
def build_scene_lines(scene_list: List[dict], timeline: List[Tuple[float, float]]) -> str:
    out = []
    for sc, (a, b) in zip(scene_list, timeline):
        loc = safe_text(sc.get("location"))
        lig = safe_text(sc.get("lighting"))
        mot = safe_text(sc.get("motion"))
        wea = safe_text(sc.get("weather"))
        moo = safe_text(sc.get("mood"))
        line = f"{a:.1f}-{b:.1f}s: location={loc}; lighting={lig}; motion={mot}; weather={wea}; mood={moo}"
        out.append(line)
    return "\n".join(out)

def build_prompt_p1(shoe_type: str, shoe_name: str, scene_list: List[dict], timeline: List[Tuple[float, float]], voice_lines: str) -> str:
    scenes_text = build_scene_lines(scene_list, timeline)
    return f"""
SORA VIDEO PROMPT - PROMPT 1 (NO CAMEO) - TOTAL 10s

Video setup:
- Vertical 9:16
- Total duration exactly 10 seconds
- Ultra sharp 4K output
- Realistic motion (not a still image)
- No on-screen text
- No logo
- No watermark
- No blur, haze, or glow

Cast rule:
- No people on screen
- No cameo
- Voice ID: {CAMEO_VOICE_ID}

Shoe reference lock:
- Use only the uploaded shoe image as reference
- Keep 100 percent shoe identity: toe shape, panels, stitching, sole, proportions
- No redesign, no deformation, no guessing, no color shift
- Lace rule: if reference shoe has laces then keep laces in all frames; if no laces then absolutely no laces

Product:
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

Scenes (inside 10s):
{scenes_text}

Audio timeline:
- 0.0-1.2s: no voice, light ambient only
- 1.2-6.9s: voice on (3 short lines)
- 6.9-10.0s: voice off, gentle fade out 9.2-10.0s

Voiceover (1.2-6.9s):
{voice_lines}
""".strip()

def build_prompt_p2(shoe_type: str, shoe_name: str, scene_list: List[dict], timeline: List[Tuple[float, float]], voice_lines: str) -> str:
    scenes_text = build_scene_lines(scene_list, timeline)
    return f"""
SORA VIDEO PROMPT - PROMPT 2 (WITH CAMEO) - TOTAL 10s

Video setup:
- Vertical 9:16
- Total duration exactly 10 seconds
- Ultra sharp 4K output
- Realistic motion (not a still image)
- No on-screen text
- No logo
- No watermark
- No blur, haze, or glow

Cast rule:
- Cameo appears naturally like a phone video review
- Cameo and Voice ID: {CAMEO_VOICE_ID}
- No strong call to action, no price, no discount, no guarantees

Shoe reference lock:
- Use only the uploaded shoe image as reference
- Keep 100 percent shoe identity: toe shape, panels, stitching, sole, proportions
- No redesign, no deformation, no guessing, no color shift
- Lace rule: if reference shoe has laces then keep laces in all frames; if no laces then absolutely no laces

Product:
- shoe_name: {shoe_name}
- shoe_type: {shoe_type}

Scenes (inside 10s):
{scenes_text}

Audio timeline:
- 0.0-1.2s: no voice, light ambient only
- 1.2-6.9s: voice on (2 short lines + 1 disclaimer line)
- 6.9-10.0s: voice off, gentle fade out 9.2-10.0s

Voiceover (1.2-6.9s):
{voice_lines}
""".strip()

# =========================
# SIDEBAR - GEMINI KEY (PERSIST)
# =========================
with st.sidebar:
    st.markdown("Gemini API Key")
    st.caption("Saved in your browser (localStorage). No need to re-enter every time.")

    api_key_input = st.text_input("GEMINI_API_KEY", value=st.session_state.gemini_api_key, type="password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save key", use_container_width=True):
            k = (api_key_input or "").strip()
            st.session_state.gemini_api_key = k
            localstorage_set_key(k, "GEMINI_API_KEY")
            st.success("Saved.")
    with c2:
        if st.button("Clear key", use_container_width=True):
            st.session_state.gemini_api_key = ""
            localstorage_clear_key("GEMINI_API_KEY")
            st.info("Cleared.")

    if st.session_state.gemini_api_key:
        st.success("Key is set.")
    else:
        st.warning("No key. AI features will fallback.")

st.divider()

# =========================
# UI
# =========================
left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload shoe image", type=["jpg", "png", "jpeg"])
    mode = st.radio("Prompt type", ["PROMPT 1 - No cameo", "PROMPT 2 - With cameo"], index=0)
    tone = st.selectbox("Voice tone", ["Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"], index=1)
    scene_count = st.slider("Number of scenes (within total 10s)", 2, 4, 3)
    count = st.slider("Number of prompts", 1, 10, 5)
    use_ai_dialogue = st.checkbox("Prefer AI dialogue (Gemini)", value=True)

with right:
    st.subheader("Notes")
    st.write("Total duration is always 10 seconds. Scenes are split inside 10 seconds.")
    st.write("Prompt 1: 3 voice lines. No disclaimer line.")
    st.write("Prompt 2: 2 voice lines plus 1 disclaimer line (total 3 lines).")
    st.caption("Dialogues columns: " + ", ".join([safe_text(x) for x in dialogue_cols]))
    st.caption("Scenes columns: " + ", ".join([safe_text(x) for x in scene_cols]))

st.divider()

if uploaded:
    shoe_name = safe_text(Path(uploaded.name).stem.replace("_", " ").strip())
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

    detect_mode = st.selectbox(
        "shoe_type mode",
        ["AI (image) - preferred", "Auto (filename) - fallback", "Manual pick"],
        index=0
    )

    detected_filename = detect_shoe_from_filename(uploaded.name)

    detected_ai = None
    raw_ai = ""
    if detect_mode.startswith("AI"):
        detected_ai, raw_ai = gemini_detect_shoe_type(img, st.session_state.gemini_api_key)

    if detect_mode.startswith("AI"):
        if detected_ai:
            shoe_type = detected_ai
            st.success(f"AI shoe_type: {shoe_type}")
            st.caption("AI raw: " + safe_text(raw_ai))
        else:
            shoe_type = detected_filename
            st.warning("AI shoe_type failed. Using filename fallback.")
            st.caption("AI raw: " + safe_text(raw_ai))
            st.info("Fallback shoe_type: " + safe_text(detected_filename))
    elif detect_mode.startswith("Auto"):
        shoe_type = detected_filename
        st.info("Filename shoe_type: " + safe_text(shoe_type))
    else:
        shoe_type = st.selectbox("Manual shoe_type", SHOE_TYPES, index=SHOE_TYPES.index("leather") if "leather" in SHOE_TYPES else 0)
        st.success("Manual shoe_type: " + safe_text(shoe_type))

    st.caption("shoe_name: " + shoe_name)

    btn_label = "Generate PROMPT 1" if mode.startswith("PROMPT 1") else "Generate PROMPT 2"
    if st.button(btn_label, use_container_width=True):
        arr = []
        for _ in range(count):
            d_pool = filter_dialogues(shoe_type, tone)
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")

            scene_list = pick_n_unique_scenes(shoe_type, scene_count)
            timeline = split_10s_timeline(scene_count)

            if mode.startswith("PROMPT 1"):
                ai_lines = None
                if use_ai_dialogue:
                    ai_lines = ai_generate_dialogue_lines(st.session_state.gemini_api_key, shoe_type, tone, n_lines=3)
                if ai_lines:
                    voice_lines = "\n".join([dot(x) for x in ai_lines])
                else:
                    voice_lines = get_dialogue_3_sentences(d, tone)

                p = build_prompt_p1(shoe_type, shoe_name, scene_list, timeline, voice_lines)

            else:
                ai_lines = None
                if use_ai_dialogue:
                    ai_lines = ai_generate_dialogue_lines(st.session_state.gemini_api_key, shoe_type, tone, n_lines=2)

                if ai_lines:
                    voice_2 = "\n".join([dot(x) for x in ai_lines])
                else:
                    voice_2 = get_dialogue_2_sentences(d, tone)

                disclaimer = random.choice(disclaimers_p2) if disclaimers_p2 else "Noi dung mang tinh chia se trai nghiem."
                disclaimer = dot(disclaimer)

                voice_lines = f"{voice_2}\n{disclaimer}"
                p = build_prompt_p2(shoe_type, shoe_name, scene_list, timeline, voice_lines)

            # Final safety normalize (avoid mojibake leaking)
            p = safe_text(p)
            arr.append(p)

        st.session_state.generated_prompts = arr

    prompts = st.session_state.get("generated_prompts", [])
    if prompts:
        st.subheader("Output prompts")
        tabs = st.tabs([str(i + 1) for i in range(len(prompts))])
        for i, tab in enumerate(tabs):
            with tab:
                st.text_area("Prompt", prompts[i], height=420, key=f"view_{i}")
                copy_button(prompts[i], key=f"copy_view_{i}")

else:
    st.warning("Upload a shoe image to begin.")

st.divider()

# Reset anti-duplicate button
if st.button("Reset anti-duplicate"):
    st.session_state.used_dialogue_ids.clear()
    st.session_state.used_scene_ids.clear()
    st.session_state.generated_prompts = []
    st.success("Reset done.")
