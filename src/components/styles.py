import streamlit as st

def apply_custom_styles():
    """커스텀 스타일 적용"""
    st.markdown("""
        <style>
        /* 전체 앱 스타일 */
        .stApp {
            background-color: #1E1E1E;
        }
        
        /* 사이드바 스타일 */
        .css-1d391kg {
            background-color: #252526;
        }
        
        /* 입력 폼 스타일 */
        .stTextInput input {
            background-color: #2D2D2D;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.8rem;
        }
        
        .stButton button {
            background-color: #007AFF;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.8rem 1.5rem;
        }
        
        /* 채팅 컨테이너 스타일 */
        .main-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            margin-bottom: 100px;
        }
        
        /* 스크롤바 스타일 */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1E1E1E;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #4A4A4A;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #5A5A5A;
        }
        </style>
    """, unsafe_allow_html=True) 