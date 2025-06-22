import torchaudio
import torch
import os
from transformers import pipeline, AutoModelForAudioClassification, AutoProcessor
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import pyaudio
import wave
import threading
import time
import streamlit as st
from datetime import datetime
from typing import Optional, Tuple 

# 음성 감정 인식 모델 설정
MODEL_NAME = "forwarder1121/ast-finetuned-model"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

class AudioRecorder:
    def __init__(self):
        self.frames = []
        self.stream = None
        self.p = None
        self._recording = False  # 스레드 안전한 내부 플래그
        
    @property
    def is_recording(self):
        return self._recording
        
    @is_recording.setter
    def is_recording(self, value):
        self._recording = value
        
    def start_recording(self):
        """녹음을 시작합니다."""
        try:
            print("[DEBUG] AudioRecorder.start_recording 시작")
            self.frames = []
            self._recording = True  # 내부 플래그 설정
            
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            self.p = pyaudio.PyAudio()
            print("[DEBUG] PyAudio 인스턴스 생성 완료")
            
            # 사용 가능한 입력 장치 확인
            info = self.p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            input_device = None
            
            for i in range(numdevices):
                device_info = self.p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    print(f"[DEBUG] 입력 장치 발견: {device_info.get('name')}")
                    input_device = i
                    break
            
            if input_device is None:
                print("[ERROR] 사용 가능한 입력 장치를 찾을 수 없습니다.")
                return
            
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=CHUNK
            )
            print("[DEBUG] 오디오 스트림 생성 완료")
            
            # 별도 레드에서 녹음
            def record():
                print("[DEBUG] 녹음 스레드 시작")
                while self.is_recording:  # 내부 플래그 사용
                    try:
                        data = self.stream.read(CHUNK, exception_on_overflow=False)
                        self.frames.append(data)
                    except Exception as e:
                        print(f"[ERROR] 녹음 중 오류 발생: {str(e)}")
                        break
                print("[DEBUG] 녹음 스레드 종료")
            
            self.record_thread = threading.Thread(target=record)
            self.record_thread.start()
            print("[DEBUG] 녹음 스레드 시작 완료")
            
        except Exception as e:
            print(f"[ERROR] 녹음 시작 중 오류 발생: {str(e)}")
            self._recording = False
            return None
    
    def stop_recording(self):
        """녹음을 중지하고 WAV 파일을 저장합니다."""
        try:
            print("[DEBUG] AudioRecorder.stop_recording 시작")
            if not self.is_recording:
                print("[DEBUG] 이미 녹음이 중지된 상태입니다.")
                return None
                
            self._recording = False  # 내부 플래그 해제
            print("[DEBUG] 녹음 상태를 False로 변경")
            
            # 녹음 스레드가 완전히 종료될 때까지 대기
            if hasattr(self, 'record_thread'):
                print("[DEBUG] 녹음 스레드 종료 대기")
                self.record_thread.join()
            
            # 스트림 정리
            if self.stream:
                print("[DEBUG] 오디오 스트림 정리 시작")
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
            print("[DEBUG] 오디오 스트림 정리 완료")
            
            # 프레임이 없으면 종료
            if not self.frames:
                print("[ERROR] 녹음된 프레임이 없습니다.")
                return None
            
            # WAV 파일로 저장
            temp_path = "temp_recording.wav"
            print(f"[DEBUG] WAV 파일 저장 시작: {temp_path}")
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b''.join(self.frames))
            print("[DEBUG] WAV 파일 저장 완료")
            
            return temp_path
            
        except Exception as e:
            print(f"[ERROR] 녹음 중지 중 오류 발생: {str(e)}")
            import traceback
            print(f"[ERROR] 상세 오류: {traceback.format_exc()}")
            return None

def predict_audio_emotion(audio_path: str) -> str:
    """음성 파일의 감정을 예측합니다."""
    try:
        # 기본 감정을 'Neutral'로 설정
        default_emotion = "Neutral"
        
        # 음성 파일 로드 시 backend 파라미터 추가
        waveform, sample_rate = torchaudio.load(
            audio_path, 
            backend="soundfile"  # 백엔드 명시
        )
        
        # 리샘플링 (필요한 경우)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # 모델 입력 준비
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
        
        # 감정 매핑
        emotion_mapping = {
            0: "Anger",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad"
        }
        
        predicted_emotion = emotion_mapping[predicted_label]
        print(f"[DEBUG] 예측된 음성 감정: {predicted_emotion}")
        return predicted_emotion
        
    except Exception as e:
        print(f"[ERROR] 음성 감정 분석 중 오류 발생: {str(e)}")
        return default_emotion

def transcribe_audio(audio_path: str) -> str:
    """
    오디오 파일을 텍스트로 변환합니다.
    
    Args:
        audio_path (str): 오디오 파일 경로
        
    Returns:
        str: 변환된 텍스트
    """
    try:
        # 먼저 Google Speech Recognition 시도
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language='ko-KR')
                print(f"[Google STT] 변환 결과: {text}")
                return text
            except sr.UnknownValueError:
                print("[Google STT] 음성을 인식할 수 없습니다. Whisper로 시도합니다.")
            except sr.RequestError:
                print("[Google STT] Google API 요청 실패. Whisper로 시도합니다.")
        
        # Google STT 실패 시 Whisper 사용
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, language='ko')
        text = result["text"].strip()
        print(f"[Whisper] 변환 결과: {text}")
        return text
        
    except Exception as e:
        print(f"[ERROR] 음성 변환 중 오류 발생: {str(e)}")
        return None

def process_recorded_audio(audio_bytes: bytes) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    실시간 녹음 bytes → (텍스트, 감정, 스트레스 확률[0~1])
    실패 시 (None, None, None) 반환
    """
    if not audio_bytes:
        return None, None, None

    tmp = f"temp_{int(time.time()*1e3)}.wav"
    try:
        # 1) bytes → WAV 저장 --------------------------------------------------
        with open(tmp, "wb") as f:
            f.write(audio_bytes)
        print(f"[INFO] WAV 저장 완료: {tmp}")

        # 2) Google STT -------------------------------------------------------
        r = sr.Recognizer()
        with sr.AudioFile(tmp) as src:
            r.adjust_for_ambient_noise(src, duration=0.2)
            audio_data = r.record(src)

        text = None
        for lang in ("ko-KR", "en-US"):
            try:
                text = r.recognize_google(audio_data, language=lang)
                if text:
                    print(f"[SUCCESS] {lang} 인식: {text}")
                    break
            except sr.UnknownValueError:
                continue
        if not text:
            print("[ERROR] STT 실패")
            return None, None, None

        # 3) 감정 분석 ---------------------------------------------------------
        try:
            emotion = get_emotion_from_gpt(text)
        except Exception as e:
            print(f"[WARN] 감정 분석 실패: {e}")
            emotion = "Neutral"

        # 4) 스트레스 예측 -----------------------------------------------------
        stress_prob = None
        try:
            stress_dict = st.session_state.chatbot_service.analyze_stress(tmp)
            if stress_dict:
                stress_prob = float(stress_dict.get("stressed", 0))
        except Exception as e:
            print(f"[WARN] 스트레스 모델 오류: {e}")

        return text, emotion, stress_prob

    except Exception as e:
        print(f"[ERROR] 오디오 처리 중 오류: {e}")
        return None, None, None

    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
                print(f"[INFO] 임시 파일 삭제: {tmp}")
            except Exception as e:
                print(f"[WARN] 임시 파일 삭제 실패: {e}")