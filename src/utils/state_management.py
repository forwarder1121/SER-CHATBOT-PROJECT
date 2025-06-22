import streamlit as st
from datetime import datetime
from src.app.constants import (
    DEFAULT_PERSONA,
    DEFAULT_EMOTION,
    WELCOME_MESSAGE_TEMPLATE
)
from src.app.config import OpenAIConfig
from src.core.services.chatbot_service import ChatbotService

def initialize_session_state(selected_persona: str = None):
    """세션 상태를 초기화합니다."""
    # ChatbotService 먼저 초기화
    if 'chatbot_service' not in st.session_state:
        st.session_state.chatbot_service = ChatbotService(OpenAIConfig())
    
    st.session_state.initialized = True
    st.session_state.selected_persona = selected_persona
    st.session_state.current_emotion = DEFAULT_EMOTION
    st.session_state.messages = []
    st.session_state.conversation_stats = {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'stress_scores': []
    }
    
    # 웰컴 메시지 추가
    if not st.session_state.messages and selected_persona:
        st.session_state.messages.append({
            'role': 'assistant',
            'content': WELCOME_MESSAGE_TEMPLATE.format(persona=selected_persona),
            'timestamp': datetime.now().strftime('%p %I:%M')
        })

def ensure_state_initialization(key: str, default_value):
    """
    특정 상태 키가 초기화되어 있는지 확인하고, 없으면 초기화합니다.
    """
    if key not in st.session_state:
        if key == 'chatbot_service':
            st.session_state[key] = ChatbotService(OpenAIConfig())
        elif key == 'conversation_stats':
            st.session_state[key] = {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'stress_scores': []
            }
        else:
            st.session_state[key] = default_value

def clear_session_state():
    """세션 상태를 초기화합니다."""
    # 기존 상태 백업
    old_chatbot = st.session_state.get('chatbot_service')  # chatbot_service 먼저 백업
    old_messages = st.session_state.get('messages', [])
    old_stats = st.session_state.get('conversation_stats', {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'stress_scores': []
    })
    
    # 세션 상태 초기화
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # 필요한 상태 복원 (chatbot_service 먼저 복원)
    if old_chatbot is None:
        st.session_state.chatbot_service = ChatbotService(OpenAIConfig())
    else:
        st.session_state.chatbot_service = old_chatbot
    st.session_state.messages = old_messages
    st.session_state.conversation_stats = old_stats