from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class OpenAIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY")
    chat_model: str = "gpt-4"
    temperature: float = 0.7