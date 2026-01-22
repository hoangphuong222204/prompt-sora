
# Sora Prompt Studio Pro — FINAL PATCH (Prompt 2 fixed)
# Prompt 2: 2 câu quảng cáo nhẹ + 1 câu miễn trừ (bắt buộc)
# Cameo xuất hiện từ đầu video cùng đôi giày
# Timing đảm bảo đủ 6.9s

# >>> FILE NÀY DÙNG THAY app_promptstudio_FINAL.py <<<

# (Phần đầu giữ nguyên như bản FINAL trước của chồng)
# ...

# =========================
# PROMPT 2 VOICE LOGIC — FIXED
# =========================

if mode == "p2":
    # 2 câu review nhẹ (không trùng Prompt 1)
    if use_ai_voice and st.session_state.gemini_api_key:
        gen_lines, dbg = gemini_generate_voice_lines(
            api_key=st.session_state.gemini_api_key,
            shoe_type=shoe_type,
            tone=tone,
            voice_style=voice_style,
            n_lines=2,
        )
        if gen_lines:
            review_lines = gen_lines
        else:
            d_pool = filter_dialogues(shoe_type, tone)
            d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
            review_lines = csv_voice_lines(d, 2)
    else:
        d_pool = filter_dialogues(shoe_type, tone)
        d = pick_unique(d_pool, st.session_state.used_dialogue_ids, "id")
        review_lines = csv_voice_lines(d, 2)

    # 1 câu miễn trừ — LUÔN THÊM CUỐI
    disc_raw = random.choice(disclaimers_p2) if disclaimers_p2 else "Nội dung chỉ mang tính chia sẻ trải nghiệm."
    disclaimer_line = short_disclaimer(disc_raw)

    # Gộp 3 dòng đúng thứ tự
    voice_lines = normalize_text(f"{review_lines}\n{disclaimer_line}")

# =========================
# CAST RULE PROMPT 2 — CAMEO TỪ ĐẦU
# =========================

cast_block = (
    "Cameo xuất hiện NGAY TỪ ĐẦU video cùng đôi giày\n"
    "Cameo cầm hoặc đứng cạnh giày tự nhiên\n"
    "Cameo nói trực diện camera như clip điện thoại\n"
    "CAMEO & VOICE ID: " + CAMEO_VOICE_ID + "\n"
    "No hard call to action, no price, no discount, no guarantees"
)

# =========================
# AUDIO TIMING PROMPT 2 — ĐỦ 6.9s
# =========================

"""
SPEECH TIMING — PROMPT 2 STRICT
0.0–0.6s: Ambient only, cameo chuẩn bị nói
0.6–3.0s: Line 1 (review nhẹ)
3.0–5.0s: Line 2 (review nhẹ)
5.0–6.9s: Line 3 (miễn trừ bắt buộc)
6.9s HARD STOP voice
7.0–10.0s: Music only, 30%
"""

# =========================
# END PATCH
# =========================
