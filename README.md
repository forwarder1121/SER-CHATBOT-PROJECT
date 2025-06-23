# SER-Chatbot

*(Speech Emotion & Stress-Aware Conversational AI)*

---

## 1. Overview

SER-Chatbot is a **Streamlit web-app** that listens to the user’s **voice** (WebRTC) or **text** input, detects

* **Emotion** (7-class text + 6-class voice)
* **Physiological stress** (2-class voice)

…and returns an **empathic GPT-4 response** spoken through one of several counselling *personas*.
The project demonstrates an end-to-end pipeline: audio capture → on-device ML inference → RAG-augmented LLM >> UI.

---

## 2. Core Features

| Category                   | Details                                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 🎤 **Audio & Text Intake** | Live microphone via WebRTC, or plain text box                                                                                              |
| 💡 **Affective Inference** | • DistilRoBERTa text-emotion (7 labels) <br> • StudentNet + Wav2Vec2 voice-stress (2 labels)<br> • AST fine-tuned voice-emotion (6 labels) |
| 👥 **Personas**            | 5 counsellor styles (professor, coach, peer-friend …) selected by query-param                                                              |
| 📑 **RAG Context**         | LangChain retrieval from Pinecone (PDF resources)                                                                                          |
| 📊 **Realtime Stats**      | Total chats, positive/negative ratio, rolling stress gauge                                                                                 |
| 🖥 **Lean Front-End**      | Single-page Streamlit; hashed-input de-duplication prevents double renders                                                                 |

---

## 3. High-Level Architecture

```mermaid
flowchart LR
    subgraph Client (Browser)
        A(Microphone / Text)-->B[Streamlit UI]
    end
    B-->C{Audio Handler}<-->|text|B
    C--wav-->D[Stress & Voice-Emotion Nets]
    B-->E[ChatEngine]
    D-->E
    E--prompt-->F[OpenAI GPT-4]
    E--embed-->G[Pinecone DB]
    F-->E-->B
```

*Single container (< 60 MB) deployable on Streamlit Cloud.*

---

## 4. Technology Stack

| Layer                        | Stack                                                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Front-End**                | Streamlit, WebRTC-JS                                                                                                               |
| **Back-End / Orchestration** | Python 3.10, LangChain                                                                                                             |
| **LLM**                      | OpenAI GPT-4 (via `langchain_openai.ChatOpenAI`)                                                                                   |
| **Voice Models**             | `forwarder1121/voice-based-stress-recognition` <br>`j-hartmann/distilroberta-base-emotion` <br>`forwarder1121/ast-finetuned-model` |
| **Vector DB**                | Pinecone                                                                                                                           |
| **CI / CD**                  | GitHub Actions → Streamlit Cloud                                                                                                   |

---

## 5. Quick-start

```bash
git clone https://github.com/forwarder1121/SER-CHATBOT-PROJECT.git
cd SER-CHATBOT-PROJECT

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# .env
echo "OPENAI_API_KEY=sk-..."           >> .env
echo "PINECONE_API_KEY=..."            >> .env
echo "PINECONE_ENVIRONMENT=us-east1-gcp" >> .env
echo "PINECONE_INDEX_NAME=ser-docs"    >> .env

streamlit run streamlit_app.py         # open http://localhost:8501
```

---

## 6. Repository Layout

```
SER-CHATBOT-PROJECT/
│
├─ streamlit_app.py         # entry-point
├─ requirements.txt
├─ .env.example
│
└─ src/
   ├─ app/                  # UI, session, routing
   ├─ core/services/        # ChatbotService, persona logic
   └─ utils/                # audio, RAG, error handling, state mgmt
```

---

## 7. Development & Contribution

1. Fork → `feature/<topic>` branch
2. Run pre-commit lint (`ruff`, `black`)
3. PR to `main` with clear description

Bug reports / ideas → [GitHub Issues](../../issues).

---

## 8. License

[MIT](LICENSE)

---

### ✨ Project Highlights

* **Audio Hash De-duplication** — solved Streamlit callback double-render bug.
* **Label Noise Fix** — converted StressID 3-class *(relax / arousal / valence)* → 2-class *(stress / non-stress)*; +17 pp F1.
* **Lightweight Deployment** — StudentNet distilled to 512-KB, full app cold-starts < 4 s.
* **Academic Output** — paper *“Lightweight Audio-Embedding-Based Stress Recognition”* in submission (KIISE 2024).

---
