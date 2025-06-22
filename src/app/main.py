import os, time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import speech_recognition as sr
import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T
from audio_recorder_streamlit import audio_recorder
from transformers import AutoModelForAudioClassification, AutoProcessor

from src.app.config import OpenAIConfig
from src.app.constants import (
    DEFAULT_EMOTION,
    DEFAULT_PERSONA,
    EMOTION_MAPPING,
    PERSONA_NAME_MAPPING,
    STRESS_WEIGHTS,
)
from src.components.chat_components import (
    render_conversation_stats,
    render_emotion_indicator,
    render_stress_indicator,
)
from src.components.message_display import apply_chat_styles, display_message
from src.core.services.chatbot_service import ChatbotService
from src.utils.state_management import (
    clear_session_state,
    ensure_state_initialization,
    initialize_session_state,
)

# ─────────────────────────  음성 감정 모델  ─────────────────────────
MODEL_NAME = "forwarder1121/ast-finetuned-model"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

# =============================================================================
# Helper
# =============================================================================
def get_emotion_from_gpt(prompt: str) -> str:
    predefined = list(EMOTION_MAPPING.values())
    pmt = (
        f'The user said: "{prompt}". '
        f"Classify it as one of {', '.join(predefined)} "
        f"and respond ONLY with the emotion word."
    )
    resp = st.session_state.chatbot_service.llm.invoke(pmt)
    emo = resp.content.strip()
    return emo if emo in predefined else DEFAULT_EMOTION


def process_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = 16_000,
    target_len: int = 16_000,
) -> Optional[torch.Tensor]:
    """mono → resample → trim/pad  (16 kHz · 1 sec)"""
    if waveform.shape[0] > 1:                       # stereo → mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.numel() == 0:
        return None
    if sample_rate != target_sr:                    # resample
        waveform = T.Resample(sample_rate, target_sr)(waveform)
    if waveform.shape[1] < target_len:              # pad
        pad = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:                                           # center-crop
        start = (waveform.shape[1] - target_len) // 2
        waveform = waveform[:, start : start + target_len]
    return waveform


def predict_audio_emotion(audio_path: str) -> str:
    waveform, sr = torchaudio.load(audio_path)
    proc = process_audio(waveform, sample_rate=sr)
    if proc is None:
        return DEFAULT_EMOTION
    inputs = processor(proc.squeeze(), sampling_rate=16_000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    idx = logits.argmax(-1).item()
    return EMOTION_MAPPING.get(idx, DEFAULT_EMOTION)


# 🔹 음성 → (text, emotion, stress_prob)
def process_recorded_audio(
    audio_bytes: bytes,
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    if not audio_bytes:
        return None, None, None

    tmp = f"tmp_{int(time.time()*1e3)}.wav"
    with open(tmp, "wb") as f:
        f.write(audio_bytes)

    try:
        rec = sr.Recognizer()
        with sr.AudioFile(tmp) as src:
            rec.adjust_for_ambient_noise(src, duration=0.2)
            audio = rec.record(src)

        text = None
        for lang in ("ko-KR", "en-US"):
            try:
                text = rec.recognize_google(audio, language=lang)
                if text:
                    break
            except sr.UnknownValueError:
                continue
        if not text:
            return None, None, None

        emotion = get_emotion_from_gpt(text)
        stress_prob = st.session_state.chatbot_service.analyze_stress(tmp)["stressed"]
        return text, emotion, stress_prob

    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def update_conversation_stats(emotion: str, stress_prob: float | None = None):
    stats = st.session_state.setdefault(
        "conversation_stats",
        {"total": 0, "positive": 0, "negative": 0, "stress_scores": []},
    )
    stats["total"] += 1
    if emotion == "Happy":
        stats["positive"] += 1
    elif emotion in ("Anger", "Disgust", "Fear", "Sad"):
        stats["negative"] += 1
    if stress_prob is not None:
        stats["stress_scores"].append(stress_prob * 100)


# =============================================================================
# Chat Flow
# =============================================================================
def handle_chat_message(prompt: str, persona: str):
    emo = get_emotion_from_gpt(prompt)
    reply = st.session_state.chatbot_service.get_response(prompt, persona)
    st.session_state.current_emotion = emo
    return emo, reply


def add_chat(role: str, content: str, emotion: str | None = None):
    st.session_state.setdefault("messages", []).append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%p %I:%M"),
            **({"emotion": emotion} if emotion else {}),
        }
    )


# =============================================================================
# Streamlit UI
# =============================================================================
def render_chat_area():
    st.title("채팅")
    apply_chat_styles()

    # ─── 이전 메시지 출력 ─────────────────────────────
    chat_box = st.container()
    with chat_box:
        for m in st.session_state.get("messages", []):
            display_message(m, persona=st.session_state.selected_persona)

    # ─── 입력 영역 ──────────────────────────────────
    c1, c2, c3 = st.columns([8, 1.2, 1.2])

    # 1) 텍스트 제출 콜백 -----------------------------
    def _handle_text_submit():
        txt = st.session_state.chat_input.strip()
        if not txt:
            return

        persona = st.session_state.selected_persona
        emo, reply = handle_chat_message(txt, persona)

        add_chat("user", txt, emo)
        add_chat("assistant", reply)
        update_conversation_stats(emo)
        st.session_state.current_emotion = emo
        st.session_state.chat_input = ""   # 입력창 초기화

        #st.rerun()                         # ← 중복 제거의 핵심

    # 1-A) text_input : 값만 입력받고 Enter 는 버튼 대신 사용
    with c1:
        st.text_input(
            "메시지를 입력",
            key="chat_input",
            label_visibility="collapsed",
        )

    # 2) 오디오 녹음 --------------------------------
    with c2:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=30.0,
            key="audio_recorder",
        )

    # 3) 전송 버튼 -----------------------------------
    with c3:
        st.button("전송", use_container_width=True, on_click=_handle_text_submit)

    # ─── Audio 처리 ─────────────────────────────────
    if audio_bytes:
        audio_hash = hash(audio_bytes)
        if audio_hash != st.session_state.get("last_audio_hash"):
            st.session_state.last_audio_hash = audio_hash

            with st.spinner("음성 처리 중..."):
                text, emo, stress = process_recorded_audio(audio_bytes)

            if text:
                persona = st.session_state.selected_persona
                reply, _ = st.session_state.chatbot_service.get_response(text, persona)

                add_chat("user", f"[음성] {text}", emo)
                add_chat("assistant", reply)
                update_conversation_stats(emo, stress)
                st.session_state.current_emotion = emo

                st.rerun()                 # ← 여기서도 동일하게 한 번만 다시 그림



def render_sidebar():
    with st.sidebar:
        st.title("감정인식 챗봇")
        if st.button("← 다른 페르소나 선택"):
            st.session_state.clear()
            st.query_params.clear()
            st.query_params["page"] = "home"
            st.rerun()

        render_emotion_indicator(st.session_state.get("current_emotion", DEFAULT_EMOTION))

        stats = st.session_state.get("conversation_stats", {})
        if (scores := stats.get("stress_scores")):
            render_stress_indicator(scores[-1] / 100)
        render_conversation_stats(stats)


def render_chat_page():
    # ── 1. URL → 페르소나 ─────────────────────────────────────────
    persona_url = st.query_params.get("persona")
    if not persona_url:                          # 파라미터 없으면 홈으로
        st.query_params["page"] = "home"
        st.rerun()
    persona = PERSONA_NAME_MAPPING.get(persona_url, DEFAULT_PERSONA)

    # ── 2. 세션 초기화 / 페르소나 변경 처리 ──────────────────────
    if (not st.session_state.get("initialized")
        or st.session_state.get("selected_persona") != persona):

        old_msgs = st.session_state.get("messages", [])  # 기존 대화 백업
        clear_session_state()
        initialize_session_state(persona)                # greeting ①

        if old_msgs:
            greet_text = st.session_state.messages[0]["content"]  # greeting ① 내용
            deduped = [m for m in old_msgs if m["content"] != greet_text]
            st.session_state.messages.extend(deduped)             # 중복 제거 후 복원

    # ── 3. URL 상태 동기화 ───────────────────────────────────────
    st.query_params.update({"page": "chat", "persona": persona_url})

    # ── 4. 화면 렌더 ─────────────────────────────────────────────
    render_sidebar()
    render_chat_area()

# =============================================================================
def main():
    st.set_page_config("감정인식 챗봇", "🤗", layout="wide")
    if "chatbot_service" not in st.session_state:
        st.session_state.chatbot_service = ChatbotService(OpenAIConfig())

    if st.query_params.get("page", "home") == "chat":
        render_chat_page()
    else:
        from src.app.home import render_home
        render_home()


if __name__ == "__main__":
    main()
