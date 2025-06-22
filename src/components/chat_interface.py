import streamlit as st
from datetime import datetime
from src.components.message_display import display_message

def render_chat_interface():
    # 전체 채팅 영역 스타일
    st.markdown("""
        <style>
        /* 채팅 컨테이너 */
        .chat-container {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 80px);
            margin: -1rem;
            padding: 1rem;
            overflow: hidden;
        }
        
        /* 메시지 영역 */
        .message-container {
            flex-grow: 1;
            overflow-y: auto;
            padding: 1rem;
            margin-bottom: 80px;  /* 입력창 높이만큼 여백 */
        }
        
        /* 입력 영역 */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 350px;  /* 사이드바 너비 + 여백 */
            right: 0;
            background-color: #2D2D2D;
            padding: 1rem;
            border-top: 1px solid #3A3B3C;
            z-index: 1000;
        }
        
        /* 입력창 스타일 */
        .stTextInput input {
            background-color: #3A3B3C;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 10px 15px;
        }
        
        /* 전송 버튼 */
        .stButton button {
            background-color: #007AFF;
            color: white;
            border-radius: 20px;
            padding: 8px 20px;
        }
        </style>
        
        <div class="chat-container">
            <div class="message-container">
    """, unsafe_allow_html=True)
    
    # 메시지 표시
    if 'messages' in st.session_state:
        for message in st.session_state.messages:
            display_message(message)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 입력 영역
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([6, 1])
        with cols[0]:
            user_input = st.text_input(
                "",
                placeholder="메시지를 입력하세요...",
                label_visibility="collapsed"
            )
        with cols[1]:
            submit = st.form_submit_button("전송")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit and user_input and user_input.strip():
        chatbot = st.session_state.chatbot_service
        emotions = chatbot.analyze_emotion(user_input)
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        response = chatbot.get_response(user_input)
        
        current_time = datetime.now().strftime('%p %I:%M')
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            
        st.session_state.messages.extend([
            {
                "role": "user",
                "content": user_input,
                "emotion": dominant_emotion,
                "timestamp": current_time
            },
            {
                "role": "assistant",
                "content": response,
                "timestamp": current_time
            }
        ])
        
        st.rerun()