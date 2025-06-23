
# SER-Chatbot 🗣️🎧  
*Speech Emotion & Stress Recognition Chatbot*

---

## 1. 프로젝트 한눈에 보기
| 구분 | 내용 |
|------|------|
| **목적** | 음성·텍스트 기반 **감정 + 스트레스 지수**를 실시간으로 분석하여, 맞춤형 공감 대화를 제공 |
| **특징** | <br>• 2-Class Stress(🆗/⚠️) 모델 직접 학습 &nbsp;pp&gt;<br>• 5종 **페르소나**(교수·멘토·친구 등) 선택&nbsp;+ RAG로 개인화<br>• Streamlit UI & WebRTC Mic → **브라우저 4초 내 기동** |
| **전체 플로 오너** | 데이터 EDA → 모델 파인튜닝 → 백엔드/API → UI → DevOps  |
| **성과** | 시스템 Demo 운영 중 · 국내 학회(한국정보기술학회 2024) **논문 투고 준비** |


---

## 2. 주요 기능

| 카테고리           | 기능                                                                        |
| -------------- | ------------------------------------------------------------------------- |
| **감정·스트레스 인식** | - 음성: AST → Wav2Vec2 임베딩 512 → StudentNet<br>- 텍스트: DistilRoBERTa 7-Class |
| **대화 엔진**      | GPT-4 + Retrieval-Augmented Generation (RAG)                              |
| **페르소나**       | 5종(교수·멘토·친구·상담가·예술치료사) 선택                                                 |
| **UI/UX**      | 실시간 Mic 입력, 감정 Badge, 스트레스 게이지, 통계                                        |
| **DevOps**     | CI/CD (GitHub Actions → Streamlit Cloud)                                  |

---

## 3. 역할 분담 

| 단계           |      난관 & 해결                                                   |
| ------------ | --------------------------------------------------------- |
| 논문 리뷰·데이터 정제 |  StressID 3-Class(이완/각성/쾌감) → **2-Class(Stress)** 재라벨링     |
| Stress 모델 학습 | Knowledge-Distillation(Teacher → Student), F1 +17 pp      |
| 백엔드·UI·배포    | * Streamlit 콜백 & 세션충돌 → “**입력 해시 + 명시적 re-render**” 패턴으로 해결 |

---

## 4. 기술 스택

| Layer             | Stack                                             |
| ----------------- | ------------------------------------------------- |
| **Frontend**      | Streamlit · Tailwind-style CSS                    |
| **Backend**       | Python 3.10                                       |
| **AI/ML**         | OpenAI GPT-4 · Whisper · HuggingFace Transformers |
| **Model Serving** | Torch + CPU Inference (메모리 < 400 MB)              |
| **Vector DB**     | Pinecone                                          |
| **DevOps**        | Docker · GitHub Actions · Streamlit Cloud         |

---

## 5. 설치 & 실행

```bash
# 1) 클론
git clone https://github.com/forwarder1121/SER-CHATBOT-PROJECT.git
cd SER-CHATBOT-PROJECT

# 2) 가상환경
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 3) 의존성
pip install -r requirements.txt

# 4) 환경변수 (.env)
cat <<EOF > .env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=...
PINECONE_INDEX_NAME=ser-chatbot
EOF

# 5) 실행
streamlit run streamlit_app.py   # <– 127.0.0.1:8501 에서 확인
```

---

## 6. 디렉터리 구조

```
SER-CHATBOT-PROJECT
├─ streamlit_app.py          # 실행 엔트리
├─ src/
│  ├─ app/                   # UI · 페이지 라우팅
│  ├─ components/            # 재사용 UI 컴포넌트
│  ├─ core/
│  │  ├─ models/             # pydantic 데이터 모델
│  │  └─ services/           # ChatbotService, persona
│  └─ utils/                 # 헬퍼 & 에러핸들링
└─ requirements.txt
```

---

## 7. 기여 가이드

1. 포크 후 브랜치 생성 `git checkout -b feature/awesome`
2. 커밋 `git commit -m "Add awesome feature"`
3. 푸시 `git push origin feature/awesome`
4. PR 생성 – 환영합니다!

---

## 8. 라이선스

MIT License

