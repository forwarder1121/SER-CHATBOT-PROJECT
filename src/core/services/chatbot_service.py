"""
chatbot_service.py
────────────────────────────────────────────────────────────────────────────
텍스트 감정 + 음성(오디오 임베딩) 기반 스트레스 분석을 동시에 처리하는
ChatbotService 구현.

• LLM            : OpenAI GPT-4 / GPT-3.5-turbo (langchain_openai.ChatOpenAI)
• 텍스트 감정    : j-hartmann/emotion-english-distilroberta-base
• 음성 스트레스   : forwarder1121/voice-based-stress-recognition  (StudentNet)
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import torch
import torchaudio
from huggingface_hub import login as hf_login
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModelForAudioClassification,
    pipeline,
)
from langchain_openai import ChatOpenAI

from src.app.constants import DEFAULT_PERSONA, PERSONA_PROMPTS
from src.utils.error_handling import handle_streamlit_errors
from src.utils.rag_utils import RAGUtils

# ──────────────── HF 모델 ID & 설정 ────────────────
_EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

_STRESS_REPO   = "forwarder1121/voice-based-stress-recognition"
_W2V_PIPELINE  = torchaudio.pipelines.WAV2VEC2_BASE          # 512-dim
_DEVICE        = torch.device("cpu")                        # ← CPU 고정
# ───────────────────────────────────────────────────


@dataclass
class ChatbotService:
    """
    - analyze_emotion(text)  → 감정 7-class 확률 dict
    - analyze_stress(wav)    → {'not_stressed': p0, 'stressed': p1}
    - get_response(...)      → LLM+RAG 응답, 참조 문서
    """

    def __init__(self, openai_config):
        # 0) HF 토큰(있으면) 로그인
        hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_API_TOKEN")
        )
        if hf_token:
            hf_login(hf_token)

        # 1) LLM
        self.llm = ChatOpenAI(
            api_key=openai_config.api_key,
            model_name=openai_config.chat_model,
            temperature=openai_config.temperature,
        )

        # 2) 텍스트 감정 분류기
        self._init_text_emotion_classifier()

        # 3) 음성 스트레스 분류기 + W2V 임베딩 모델
        self._init_stress_classifier()

        # 4) RAG
        self._init_rag()

    # ───────────────────────── 텍스트 감정 ─────────────────────────
    def _init_text_emotion_classifier(self) -> None:
        """DistilRoBERTa 감정 분류기 (CPU 로 완전 로드)"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tok = AutoTokenizer.from_pretrained(_EMOTION_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(
            _EMOTION_MODEL_ID,
            torch_dtype="float32",
            low_cpu_mem_usage=False,      # 메타-텐서 방지
        ).to(_DEVICE).eval()

        self.emotion_classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tok,
            top_k=None,
            device=-1,                   # 내부 .to() 호출 차단
        )

    @handle_streamlit_errors
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        results = self.emotion_classifier(text)
        return {x["label"]: x["score"] for x in results[0]}

    # ───────────────────────── 음성 스트레스 ──────────────────────
    def _init_stress_classifier(self) -> None:
        cfg = AutoConfig.from_pretrained(_STRESS_REPO, trust_remote_code=True)

        # StudentNet (2-class)
        self.stress_model = AutoModelForAudioClassification.from_pretrained(
            _STRESS_REPO,
            config=cfg,
            trust_remote_code=True,
            torch_dtype="float32",
            device_map={"": _DEVICE},
        ).eval()

        # Wav2Vec2 (512-dim 임베딩)
        bundle = _W2V_PIPELINE
        self.w2v = bundle.get_model().to(_DEVICE).eval()
        self.resample = torchaudio.transforms.Resample(
            orig_freq=bundle.sample_rate, new_freq=16_000
        )

    @handle_streamlit_errors
    def analyze_stress(self, wav_path: str) -> Optional[Dict[str, float]]:
        """
        WAV → {"stressed": p, "not_stressed": q}
        """
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0, keepdim=True)
        if sr != 16_000:
            wav = self.resample(wav)

        with torch.inference_mode():
            feats_list, _ = self.w2v.extract_features(wav.to(_DEVICE))
            if not feats_list:
                return None

            feats = feats_list[-1]               # (1, T, 768)
            if feats.size(-1) > 512:             # 🔹 768 → 512
                feats = feats[..., :512]

            emb = feats.mean(dim=1)              # (1, 512)

            logits = self.stress_model(emb).logits
            probs  = torch.softmax(logits, dim=-1)[0]

        return {
            "not_stressed": float(probs[0]),
            "stressed":     float(probs[1]),
        }

    # ─────────────────────────── RAG ────────────────────────────
    def _init_rag(self) -> None:
        try:
            self.rag_utils = RAGUtils()
            self.rag_enabled = True
            print("✅ RAG enabled")
        except Exception as e:
            print(f"⚠️  RAG init failed: {e}")
            self.rag_enabled = False

    # ──────────────────────── Chat 응답 ─────────────────────────
    @handle_streamlit_errors
    def get_response(
        self, user_input: str, persona_name: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        persona_prompt = PERSONA_PROMPTS.get(
            persona_name, PERSONA_PROMPTS[DEFAULT_PERSONA]
        )

        augmented_prompt, reference_docs = "", []
        if self.rag_enabled:
            try:
                augmented_prompt, reference_docs = self.rag_utils.get_augmented_prompt(
                    user_input, persona_name
                )
            except Exception:
                pass

        prompt = f"""
        {persona_prompt}

        {augmented_prompt}

        사용자 메시지: {user_input}

        위 페르소나 설정을 바탕으로 사용자의 메시지에 공감하고 적절한 응답을 해주세요.
        응답은 반드시 한국어로 해주세요.
        응답에는 페르소나의 특징이 잘 드러나야 합니다.
        응답에는 '페르소나의 메시지:', '챗봇의 메시지:' 등의 접두어를 포함하지 마세요.
        """

        response = self.llm.invoke(prompt)
        return response.content.strip(), reference_docs
