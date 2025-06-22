import streamlit as st
from datetime import datetime
from src.app.config import OpenAIConfig
from src.app.constants import DEFAULT_PERSONA

def initialize_chatbot_service():
    """ChatbotService 초기화"""
    from src.core.services.chatbot_service import ChatbotService
    return ChatbotService(OpenAIConfig())

def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if not st.session_state.initialized:
        # Chat state
        st.session_state.messages = []
        st.session_state.last_message = None
        st.session_state.current_emotion = "Neutral"
        st.session_state.selected_persona = "박준영 멘토"
        
        # 녹음 관련 상태 추가
        st.session_state.is_recording = False
        st.session_state.audio_recorder = None
        
        # Conversation stats
        st.session_state.conversation_stats = {
            'total': 0,
            'positive': 0,
            'negative': 0
        }
        
        # Initialize chatbot service
        st.session_state.chatbot_service = initialize_chatbot_service()
        
        st.session_state.initialized = True