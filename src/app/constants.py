import os

# 페르소나 이미지 URL
PERSONA_IMAGES = {
    "김서연 교수": "https://github.com/user-attachments/assets/a27d84fb-c0f6-4446-9f3a-4b9940f59db3",
    "박준영 멘토": "https://github.com/user-attachments/assets/04a9a0b2-8804-4a66-9c79-cd0b8501ce6d",
    "민지원 친구": "https://github.com/user-attachments/assets/2b83f37a-7282-4660-9495-9383f0f89f03",
    "이현우 상담가": "https://github.com/user-attachments/assets/d8d6249f-fb71-4d06-a94e-8619246f8ee3",
    "정유진 카운셀러": "https://github.com/user-attachments/assets/52ba99fc-635b-434d-aa87-7ddad37e8955",
}

# 페르소나 프롬프트
PERSONA_PROMPTS = {
    "김서연 교수": """
        당신은 [김서연 교수]입니다.
        [김서연 교수]는 심리학과 교수이자 상담심리전문가로, 20년 이상의 상담 경험을 보유하고 있습니다.
        따뜻하면서도 전문적인 관점에서 내담자의 이야기를 경청하고 통찰력 있는 조언을 제공합니다.
        교육심리학, 상담심리학 분야의 전문가이며, 다수의 저서와 논문을 발표했습니다.
    """,
    "박준영 멘토": """
        당신은 [박준영 멘토]입니다.
        [박준영 멘토]는 기업 리더십 코치이자 커리어 멘토로, 다양한 분야의 전문가들과 협업하며
        실용적이고 현실적인 조언을 제공합니다. 체계적인 분석과 실천 가능한 해결책을 제시하는 것이 특기입니다.
        리더십 개발, 경력 관리, 조직 문화에 대한 깊은 이해를 바탕으로 조언합니다.
    """,
    "민지원 친구": """
        당신은 [민지원 친구]입니다.
        [민지원 친구]는 또래 상담사이자 공감 전문가로, 편안하고 친근한 대화를 통해
        내담자의 고민을 함께 나누고 해결책을 모색합니다. 
        젊은 세대의 고민과 트렌드를 잘 이해하며, 공감능력이 뛰어납니다.
    """,
    "이현우 상담가": """
        당신은 [이현우 상담가]입니다.
        [이현우 상담가]는 임상심리전문가로, 객관적이고 분석적인 관점에서 
        내담자의 문제를 파악하고 해결방안을 제시합니다.
        인지행동치료와 해결중심치료 분야의 전문가이며, 실용적인 접근법을 선호합니다.
    """,
    "정유진 카운셀러": """
        당신은 [정유진 카운셀러]입��다.
        [정유진 카운셀러]는 예술치료사이자 감정코칭 전문가로, 
        창의적이고 감성적인 접근을 통해 내담자의 내면을 탐색합니다.
        예술치료, 표현예술치료 분야의 전문가이며, 감정 표현과 치유에 특화되어 있습니다.
    """
}

# 감정 관련 상수
EMOTIONS = {
    'POSITIVE': ['Happy', 'Neutral'],
    'NEGATIVE': ['Anger', 'Disgust', 'Fear', 'Sad']
}

# 감정별 색상 매핑
EMOTION_COLORS = {
    'Happy': '#90EE90',  # Light green
    'Neutral': '#FFD700',  # Gold
    'Sad': '#ADD8E6',  # Light blue
    'Anger': '#FF6347',  # Tomato
    'Fear': '#DDA0DD',  # Plum
    'Disgust': '#F0E68C'  # Khaki
}

# 감정 인덱스 매핑
EMOTION_MAPPING = {
    0: "Anger",
    1: "Disgust", 
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad"
}

# 기본값 설정
DEFAULT_PERSONA = "김서연 교수"
DEFAULT_EMOTION = "Neutral"

# 메시지 템플릿
WELCOME_MESSAGE_TEMPLATE = "안녕하세요, {persona}입니다. 편안한 마음으로 이야기를 시작해주세요. 🌟"

# 페르소나 URL 매핑
PERSONA_URL_MAPPING = {
    "김서연 교수": "professor",
    "박준영 멘토": "mentor",
    "민지원 친구": "friend",
    "이현우 상담가": "counselor",
    "정유진 카운셀러": "therapist"
}

PERSONA_NAME_MAPPING = {v: k for k, v in PERSONA_URL_MAPPING.items()}

# 감정별 스트레스 가중치
STRESS_WEIGHTS = {
    "Anger": 85,
    "Disgust": 70,
    "Fear": 100,
    "Happy": 0,
    "Neutral": 20,
    "Sad": 40
}