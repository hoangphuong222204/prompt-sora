import streamlit as st
import pandas as pd
import random
import io

# =========================
# 🔹 ĐỌC DỮ LIỆU TỪ CSV
# =========================
dialogues = pd.read_csv("dialogue_library.csv").to_dict(orient="records")
scenes = pd.read_csv("scene_library_900.csv").to_dict(orient="records")

used_dialogues = set()
used_scenes = set()

def pick_unique_random(pool, used):
    """Chọn ngẫu nhiên không trùng lặp"""
    choices = [x for x in pool if x not in used]
    if not choices:
        used.clear()
        choices = pool.copy()
    choice = random.choice(choices)
    used.add(choice)
    return choice


# =========================
# 🔹 HÀM TẠO PROMPT CHI TIẾT
# =========================
def tao_prompt_unique(shoe_type, has_cameo):
    scene = pick_unique_random([s['scene'] for s in scenes], used_scenes)
    dialogue = pick_unique_random([d['dialogue'] for d in dialogues], used_dialogues)
    tone = random.choice(['Tự nhiên', 'Mạnh mẽ', 'Truyền cảm', 'Lãng mạn', 'Tự tin'])
    style = random.choice(['Luxury', 'Street', 'Nature', 'Rain', 'Studio', '3D', 'Sport'])

    cameo_text = "@phuongnghi18091991" if has_cameo else "Voice cameo (ẩn nhân vật)"

    prompt = f"""
🎬 **SORA PROMPT STUDIO PRO – 4K HDR**

[Product Type]: {shoe_type.upper()}
[Style]: {style}
[Scene]: {scene}

[Camera Motion]: Orbit 360°, dolly-in/out tự nhiên, ánh sáng rõ logo, không đảo chữ.
[Voiceover {cameo_text} | Tone {tone} | 0–6.9s]: {dialogue}
[Music]: Nhạc nền phù hợp tone {tone}, fade-out tự nhiên lúc 9–10s.
[Quality]: 4K HDR, ánh sáng trung thực, clarity lock, không noise.
[Safety]: Không logo đảo, không text/link, không vi phạm chính sách TikTok Shop.
"""
    return prompt


# =========================
# 🔹 GIAO DIỆN STREAMLIT
# =========================
st.set_page_config(page_title="Sora Prompt Studio Pro – 4K HDR", layout="wide")
st.title("🎥 SORA PROMPT STUDIO PRO – AI PROMPT GENERATOR")

uploaded_file = st.file_uploader("📸 Tải ảnh giày hoặc dép", type=["jpg", "jpeg", "png"])

prompt_type = st.radio("🎭 Chọn loại prompt", ["Prompt 1 – Không cameo", "Prompt 2 – Có cameo"])
so_luong = st.slider("📦 Số lượng prompt muốn tạo", 1, 10, 5)

shoe_type = st.selectbox(
    "👟 Loại giày nhận dạng:",
    ["sneaker", "loafer", "sandals", "boot", "slide", "flipflop"],
)

if uploaded_file:
    st.image(uploaded_file, caption="Ảnh mẫu đã tải lên", use_column_width=True)

if st.button("✨ Sinh Prompt Chi Tiết (Tự Động Nhiều Mẫu)"):
    prompts = []
    for i in range(so_luong):
        p = tao_prompt_unique(shoe_type, prompt_type == "Prompt 2 – Có cameo")
        prompts.append(p)

    st.success(f"✅ Đã tạo {so_luong} prompt chi tiết không trùng lặp.")
    for i, p in enumerate(prompts):
        st.text_area(f"🎬 Prompt {i+1}", p, height=270)
        st.button(f"📋 Sao chép Prompt {i+1}", key=f"copy_{i}")

    # Xuất CSV tải xuống
    df = pd.DataFrame(prompts, columns=["Prompt"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Tải tất cả prompt (.csv)", csv, "prompts.csv", "text/csv")
