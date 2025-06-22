import streamlit as st

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("""
            <h1 style='
                color: white;
                margin-bottom: 2rem;
            '>감정인식 챗봇 🏠</h1>
        """, unsafe_allow_html=True)
        
        st.markdown("### 사용 방법")
        st.markdown("""
            1. 채팅창에 현재 기분이나 상황을 입력하세요.
            2. 음성 파일을 업로드하여 감정을 분석할 수 있습니다.
            3. 챗봇이 감정을 분석하고 공감적인 대화를 제공합니다.
            4. 필요한 경우 적절한 조언이나 위로를 받을 수 있습니다.
        """)
        
        # 현재 선택된 페르소나 표시 
        if 'selected_persona' in st.session_state:
            st.markdown(f"### 현재 대화 상대: {st.session_state.selected_persona}")
        
        # 현재 감정 상태 표시
        if 'current_emotion' in st.session_state:
            st.markdown(f"### {st.session_state.current_emotion}")
        
        # 대화 통계
        st.markdown("### 대화 통계")
        stats = st.session_state.get('conversation_stats', {})
        st.write(f"총 대화 수: {stats.get('total', 0)}")
        st.write(f"긍정적 감정: {stats.get('positive', 0)}")
        st.write(f"부정적 감정: {stats.get('negative', 0)}")
        
        # 참고 문서 섹션
        st.markdown("### 📚 참고 문서")
        if 'messages' in st.session_state:
            # 디버그 출력 추가
            print("\n=== Checking Messages for Documents ===")
            print(f"Number of messages: {len(st.session_state.messages)}")
            
            # 가장 최근 메시지의 참고 문서 표시
            latest_docs = None
            for msg in reversed(st.session_state.messages):
                print(f"Checking message: {msg}")
                if msg.get('role') == 'assistant' and msg.get('reference_docs'):
                    latest_docs = msg['reference_docs']
                    print(f"Found docs: {latest_docs}")
                    break
            
            if latest_docs:
                for doc in latest_docs:
                    disease = doc['metadata'].get('disease', '')
                    tab = doc['metadata'].get('tab', '')
                    content = doc.get('content', '').strip()
                    
                    with st.expander(f"📑 {disease} - {tab}"):
                        st.markdown(f"""
                            <div style='
                                background-color: #2d2d2d;
                                padding: 1rem;
                                border-radius: 0.5rem;
                                margin-bottom: 0.5rem;
                                color: white;
                            '>
                                {content}
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("아직 참고한 문서가 없습니다.")
        
        # 음성 파일 업로드 섹션
        st.markdown("### 음성 파일 업로드")
        st.markdown("지원 형식: WAV")
        
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=['wav'],
            help="Limit 200MB per file • WAV"
        )