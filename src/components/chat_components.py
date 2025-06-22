# src/components/chat_components.py
import streamlit as st
from src.app.constants import EMOTION_COLORS, PERSONA_IMAGES


# ───────────────────────── 감정 배지 ─────────────────────────
def render_emotion_indicator(emotion: str):
    color = EMOTION_COLORS.get(emotion, EMOTION_COLORS["Neutral"])
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin-top:16px;">
            <span style="
                background:{color};
                color:white;
                padding:4px 12px;
                border-radius:12px;
                font-weight:600;">
                {emotion}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────── 대화 통계 표 ───────────────────────
def render_conversation_stats(stats: dict):
    st.markdown("### 대화 통계")
    st.write(f"- 총 대화 수&nbsp;: {stats.get('total', 0)}")
    st.write(f"- 긍정적 감정&nbsp;: {stats.get('positive', 0)}")
    st.write(f"- 부정적 감정&nbsp;: {stats.get('negative', 0)}")


# ──────────────────────── 메시지 출력 ───────────────────────
def display_message(msg: dict, *, persona: str):
    """msg = {'role': 'user'|'assistant', 'content': str, 'emotion': ?, 'timestamp': ?}"""
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant", avatar=PERSONA_IMAGES.get(persona)):
            st.write(msg["content"])


# ─────────────────────── 스트레스 인디케이터 ───────────────────────
def render_stress_indicator(stress: float):
    """
    stress : 0~1 확률 또는 0~100 점수 (자동 보정)
    0‧‧‧30 green / 30‧‧‧60 orange / 60‧‧‧ red
    """
    if stress <= 1.0:
        stress *= 100
    if stress < 30:
        color, level = "green", "낮음"
    elif stress < 60:
        color, level = "orange", "보통"
    else:
        color, level = "red", "높음"

    st.markdown(
        f"""
        <div style='padding:10px;border-radius:5px;background:{color}20;'>
            <p style='color:{color};font-size:20px;margin:0;'>
                {stress:,.1f} / 100&nbsp;({level})
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
