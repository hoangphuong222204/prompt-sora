import streamlit as st
import random
import pandas as pd

# ============== CẤU HÌNH ỨNG DỤNG =================
st.set_page_config(page_title="Sora Prompt Studio Pro – 4K HDR Việt Nam Edition", layout="centered")

st.markdown("<h2 style='color:#1976D2;text-align:center;'>🎬 Sora Prompt Studio Pro – 4K HDR Việt Nam Edition</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>@phuongnghi18091991 Studio</p>", unsafe_allow_html=True)

# ============== THANH MENU =================
menu = st.radio("Chọn chế độ:", ["Prompt 1 – Không cameo", "Prompt 2 – Có cameo @phuongnghi18091991"])

uploaded_file = st.file_uploader("📤 Tải ảnh giày/dép (tùy chọn, giúp AI chọn phong cách):", type=["jpg", "png", "jpeg"])
so_prompt = st.slider("Số lượng prompt cần tạo:", 1, 10, 3)

st.markdown("---")

# ============== DỮ LIỆU PROMPT MẪU =================
p1_templates = [
    """🎬 PROMPT 1 – GIÀY THỂ THAO TRẮNG (4K HDR)
[Scene] Studio nền trắng tinh, ánh sáng 45° từ trái, giày lơ lửng giữa khung, logo rõ nét.
[Camera] 0–3s: Orbit chậm quanh giày. 3–6s: Zoom-in chi tiết logo. 6–10s: Dolly-out toàn cảnh.
[Voiceover – @phuongnghi18091991] “Phong cách thật đến từ điều giản dị.”
[Quality] Ultra 4K HDR, clarity lock, anti-flip, ánh sáng vật lý thật, không noise.
""",
    """🎬 PROMPT 1 – GIÀY DA NÂU LUXURY (4K HDR)
[Scene] Nền đen sang trọng, ánh sáng phản kim loại vàng nhẹ, phản sáng bóng mờ.
[Camera] 0–2s: Pan chậm ngang thân giày. 2–5s: Orbit nửa vòng, focus logo. 5–10s: Zoom-out ánh sáng mượt.
[Voiceover – @phuongnghi18091991] “Tôi chọn tinh tế thay vì ồn ào.”
[Quality] 4K HDR, ánh sáng trung tính, không lật logo, phản chiếu thực tế.
"""
]

p2_templates = [
    """🎬 PROMPT 2 – CAMEO @phuongnghi18091991 NGOÀI TRỜI (4K HDR)
[Scene] Sáng sớm, ánh sáng vàng nhẹ, cameo bước chậm trên đường đá, mang giày trắng.
[Camera] 0–2s: Focus đôi giày khi bước. 2–6s: Dolly-in toàn thân cameo. 6–10s: Orbit 3/4 hậu cảnh.
[Voiceover – @phuongnghi18091991] “Mỗi ngày là một cơ hội để bắt đầu tốt hơn.”
[Quality] HDR tone vàng sáng, motion mượt, không méo hình, logo rõ nét.
""",
    """🎬 PROMPT 2 – CAMEO @phuongnghi18091991 STUDIO (4K HDR)
[Scene] Studio đen trắng, cameo chỉnh dây giày dưới ánh đèn spotlight phản sáng nhẹ.
[Camera] 0–3s: Close-up giày. 3–6s: Pan lên khuôn mặt cameo. 6–10s: Zoom-out toàn cảnh fade trắng.
[Voiceover – @phuongnghi18091991] “Tôi không chọn nổi bật – tôi chọn tinh tế.”
[Quality] Ultra 4K HDR, depth of field thật, ánh sáng vật lý chính xác, không lật chữ.
"""
]

# ============== XỬ LÝ TẠO PROMPT =================
def tao_prompt(loai, so_luong):
    prompts = []
    templates = p1_templates if loai == 1 else p2_templates
    for i in range(so_luong):
        prompts.append(random.choice(templates))
    return prompts

# ============== NÚT SINH PROMPT =================
if st.button("▶️ Sinh Prompt"):
    loai = 1 if "Không cameo" in menu else 2
    prompts = tao_prompt(loai, so_prompt)
    st.success(f"Đã tạo {so_prompt} prompt ({menu})")

    for i, p in enumerate(prompts, 1):
        st.text_area(f"Prompt {i}", p, height=250)
        st.button(f"📋 Sao chép Prompt {i}", key=f"copy_{i}")

    # Xuất CSV
    df = pd.DataFrame({"Prompt": prompts})
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📄 Tải tất cả Prompt (CSV)", csv, "prompts.csv", "text/csv")

# ============== HIỂN THỊ ẢNH TẢI LÊN =================
if uploaded_file:
    st.image(uploaded_file, caption="Ảnh đã tải lên", use_column_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2026 @phuongnghi18091991 Studio – Sora Prompt Studio Pro 4K HDR</p>", unsafe_allow_html=True)
