# EmotiLens

> **Note:** The original repository's model/ folder (LLM client, predictor, explainer) was not published, so these modules were reimplemented in this version.

## Features
- Emotion Detection (GoEmotions)
- Explainable word-level importance (occlusion analysis)
- Chatbot (LLM)
- Exercise Recommendations
- Book Recommendations (Google Books API)

## Tech Stack
- FastAPI
- Node.js frontend
- HuggingFace Transformers
- Ollama / LLM
- AWS (planned)

## Setup
1. Clone repo
2. Add .env file
3. Run backend:
   uvicorn app.main:app --reload

## Contributions
- Explainable emotion detection using occlusion-based word importance
- Context-aware chatbot responses
- Hybrid recommendation system (LLM + API)
- Secure backend API integration
- Emotion-driven exercise suggestions
