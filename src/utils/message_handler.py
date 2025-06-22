import streamlit as st
from datetime import datetime

def handle_message_submission(prompt: str) -> None:
    """사용자 메시지를 처리하고 챗봇의 응답을 생성하는 함수"""
    if not prompt or not prompt.strip():
        return
    
    # 중복 체크를 위한 키 생성
    message_key = f"{prompt}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    if 'processed_messages' not in st.session_state:
        st.session_state.processed_messages = set()
    
    # 이미 처리된 메시지인지 확인
    if message_key in st.session_state.processed_messages:
        return
        
    try:
        chatbot = st.session_state.get('chatbot_service')
        if chatbot:
            # 응답 생성
            response, reference_docs = chatbot.get_response(prompt)
            
            # 디버그 출력 추가
            print("\n=== Saving Message ===")
            print(f"Response: {response}")
            print(f"Reference docs: {reference_docs}")
            
            # 감정 분석 및 응답 생성
            emotions = chatbot.analyze_emotion(prompt)
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # 메시지 목록 초기화
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # 새 메시지 추가
            current_time = datetime.now().strftime('%p %I:%M')
            st.session_state.messages.extend([
                {
                    "role": "user",
                    "content": prompt,
                    "timestamp": current_time,
                    "emotion": dominant_emotion
                },
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": current_time,
                    "reference_docs": reference_docs
                }
            ])
            
            # 디버그 출력
            print(f"Messages after update: {st.session_state.messages}")
            
            # 처리된 메시지 기록
            st.session_state.processed_messages.add(message_key)
            st.session_state.current_emotion = dominant_emotion
            
            # 대화 통계 업데이트
            if 'conversation_stats' not in st.session_state:
                st.session_state.conversation_stats = {'total': 0, 'positive': 0, 'negative': 0}
            
            st.session_state.conversation_stats['total'] += 1
            if dominant_emotion in ['joy', 'love', 'surprise']:
                st.session_state.conversation_stats['positive'] += 1
            elif dominant_emotion in ['anger', 'sadness', 'fear']:
                st.session_state.conversation_stats['negative'] += 1
            
        else:
            st.error("챗봇 서비스가 초기화되지 않았습니다.")
            
    except Exception as e:
        st.error(f"메시지 처리 중 오류가 발생했습니다: {str(e)}")