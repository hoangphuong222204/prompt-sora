import streamlit as st
import random
import pandas as pd
import random

# Đọc dữ liệu thoại và bối cảnh
dialogues = pd.read_csv("dialogue_library.csv").to_dict(orient="records")
scenes = pd.read_csv("scene_library.csv").to_dict(orient="records")

# Bộ nhớ tạm để tránh trùng lặp
used_dialogues = set()
used_scenes = set()

def pick_unique_random(pool, used):
    choices = [x for x in pool if x not in used]
    if not choices:  # reset khi hết
        used.clear()
        choices = pool.copy()
    choice = random.choice(choices)
    used.add(choice)
    return choice

st.set_page_config(page_title="Sora Prompt Studio Pro – Director Edition", layout="wide")
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Tạo Prompt",
    "🎙️ Thoại Cameo",
    "🛡️ Kiểm tra an toàn",
    "🎞️ Ghép cảnh"
])
def nhan_dien_giay(ten_file):
    if "da" in ten_file.lower():
        return "leather"
    elif "sandal" in ten_file.lower() or "dep" in ten_file.lower():
        return "sandals"
    elif "run" in ten_file.lower() or "thethao" in ten_file.lower():
        return "runner"
    elif "boot" in ten_file.lower():
        return "boots"
    elif "lux" in ten_file.lower():
        return "luxury"
    elif "casual" in ten_file.lower():
        return "casual"
    return "sneaker"


def chon_phong_cach_va_tone(shoe_type, has_cameo):
    styles_no_cameo = ["A1","A2","A3","A4","A5","A6","A7"]
    styles_with_cameo = ["B1","B2","B3","B4","B5","B6","B7","B8"]
    mapping = {
        "sneaker": ("A6", "Tự tin") if not has_cameo else ("B1", "Tự tin"),
        "leather": ("A2", "Truyền cảm") if not has_cameo else ("B2", "Truyền cảm"),
        "sandals": ("A3", "Tự nhiên") if not has_cameo else ("B4", "Tự nhiên"),
        "runner": ("A5", "Tự tin") if not has_cameo else ("B5", "Mạnh mẽ"),
        "boots": ("A4", "Mạnh mẽ") if not has_cameo else ("B6", "Tự tin"),
        "casual": ("A7", "Lãng mạn") if not has_cameo else ("B7", "Lãng mạn"),
        "luxury": ("A2", "Truyền cảm") if not has_cameo else ("B8", "Truyền cảm")
    }
    return mapping.get(shoe_type, random.choice(styles_with_cameo if has_cameo else styles_no_cameo))


def sinh_thoai(tone):
    thu_vien = {
        "Truyền cảm": [
            "Mỗi bước đi là một lời kể không cần nói ra.",
            "Phong cách thật đến từ những điều giản dị nhất.",
            "Tôi chọn sự tinh tế trong từng chi tiết."
        ],
        "Tự tin": [
            "Tôi không đợi cơ hội – tôi tạo ra cơ hội trong từng bước.",
            "Tôi đi theo cách riêng của mình.",
            "Bản lĩnh là khi bạn dám khác biệt."
        ],
        "Mạnh mẽ": [
            "Không có gì có thể làm tôi dừng lại.",
            "Mỗi vết bẩn là một dấu ấn của hành trình.",
            "Tôi chọn đi, thay vì đứng yên."
        ],
        "Lãng mạn": [
            "Giữa hoàng hôn này, tôi bước cùng cảm xúc.",
            "Mỗi hơi thở, mỗi nhịp tim – một câu chuyện.",
            "Tôi tìm thấy chính mình trong từng bước đi."
        ],
        "Tự nhiên": [
            "Không cần cố gắng để nổi bật – chỉ cần là chính mình.",
            "Mọi thứ xung quanh đều đang thở cùng tôi.",
            "Tôi lặng yên, nhưng không dừng lại."
        ]
    }
    return "\n".join(random.sample(thu_vien.get(tone, []), 3))


def tao_prompt_unique(shoe_type, has_cameo):
    # Chọn tone phù hợp
    tones = ["Tự tin","Truyền cảm","Mạnh mẽ","Lãng mạn","Tự nhiên"]
    tone = random.choice(tones)

    # Lọc dữ liệu theo tone và loại giày
    dialogue_pool = [d["text"] for d in dialogues if d["tone"] == tone and d["shoe_type"] == shoe_type]
    scene_pool = [f"{s['lighting']}, {s['location']}, {s['motion']}, {s['weather']}, {s['mood']}" for s in scenes if s["shoe_type"] == shoe_type]

    # Nếu không tìm thấy, fallback toàn bộ tone
    if not dialogue_pool: dialogue_pool = [d["text"] for d in dialogues if d["tone"] == tone]
    if not scene_pool: scene_pool = [f"{s['lighting']}, {s['location']}, {s['motion']}, {s['weather']}, {s['mood']}" for s in scenes]

    # Lấy thoại & cảnh không trùng
    dialogue = pick_unique_random(dialogue_pool, used_dialogues)
    scene = pick_unique_random(scene_pool, used_scenes)

    cameo = "@phuongnghi18091991" if has_cameo else "Voice cameo only"

    return f"""
🎬 PROMPT {'2' if has_cameo else '1'} – {cameo} | {shoe_type.upper()} | Tone {tone}

[Scene] {scene}

[Voiceover – {cameo} | 6.9s]
{dialogue}

[Music] Nhạc nền {tone.lower()}, fade-out tự nhiên 6.9–10s.  
[Quality] 4K HDR, không logo, không text, đúng chính sách TikTok Shop.
"""


with tab1:
    st.header("prompt = tao_prompt_unique(shoe_type, has_cameo)")
    uploaded_file = st.file_uploader("Tải ảnh giày/dép", type=["jpg","png"])
    has_cameo = st.radio("Chọn loại prompt", [
        "Prompt 1 – Không cameo", 
        "Prompt 2 – Có cameo"
    ]) == "Prompt 2 – Có cameo"

    so_luong = st.slider("Số lượng prompt muốn tạo", 1, 10, 5)
    st.caption("💡 Mặc định app sẽ sinh 5 prompt chi tiết khác nhau cho cùng sản phẩm.")

    if uploaded_file:
        shoe_type = nhan_dien_giay(uploaded_file.name)
        st.write(f"👟 Loại giày nhận dạng: **{shoe_type}**")

        if st.button("🎬 Sinh Prompt Chi Tiết (Tự Động 5 Mẫu)"):
            prompts = []
            for i in range(so_luong):
                prompt = tao_prompt(shoe_type, has_cameo)
                prompts.append(prompt)
                st.markdown(f"### 🎞️ Prompt {i+1}")
                st.text_area(f"Prompt chi tiết {i+1}", prompt, height=400, key=f"prompt_{i}")
                st.button(f"📋 Sao chép Prompt {i+1}", key=f"copy_{i}")
            
            st.success(f"✅ Đã tạo {so_luong} prompt chi tiết. Hãy chọn prompt phù hợp nhất và dán vào Sora.")

with tab2:
    st.header("🎙️ Tạo thoại Cameo")
    tone = st.selectbox("Chọn tone thoại", [
        "Truyền cảm", "Tự tin", "Mạnh mẽ", "Lãng mạn", "Tự nhiên"
    ])
    if st.button("🎤 Sinh Thoại"):
        st.text_area("Thoại 3 câu (6.9s):", sinh_thoai(tone), height=150)
with tab3:
    st.header("🛡️ Kiểm tra an toàn TikTok Shop")
    txt = st.text_area("Nhập prompt để kiểm tra:", height=200)
    if st.button("🔍 Kiểm tra"):
        vi_pham = [t for t in ["link","giá","QR","STD","giảm","mua ngay"] if t in txt.lower()]
        if vi_pham:
            st.error(f"⚠️ Phát hiện từ cấm: {', '.join(vi_pham)}")
        else:
            st.success("✅ Không phát hiện nội dung vi phạm.")
with tab4:
    st.header("🎞️ Ghép cảnh A–B")
    st.write("Bạn có thể ghép Prompt 1 + Prompt 2 thành video 20s bằng Google Colab.")
    st.markdown("[Mở hướng dẫn ghép video trên Colab](https://colab.research.google.com)")
