# Local LLM Demo

Simple FastAPI + Streamlit demo for surgical tool usage prediction.

## Architecture
- **FastAPI (uvicorn)** exposes the prediction API backed by a scikit-learn pipeline serialized to `model.pkl`.
- **Streamlit** provides the UI and calls the FastAPI service for predictions, then calls Llama for explanations.
- **Ollama (Llama 3.2)** serves the local LLM over HTTP (port 11434) for explanations.
- **Docker Compose** orchestrates all three: `api` (8000), `streamlit` (8501), and `ollama` (11434). Streamlit uses `PREDICT_URL` to reach the API and `OLLAMA_URL` to reach Ollama.

### Dependency flow (ASCII)
```
[User Browser]
      |
      v
 [Streamlit UI] --(HTTP: predict)--> [FastAPI / Uvicorn]
      |                                   |
      |                                   v
      |                              [model.pkl]
      |
      +--(HTTP: generate, 11434)--> [Ollama Llama 3.2]
```

## Setup
- Python 3.10+ recommended.
- Create venv and install deps:
  - `python -m venv .venv`
  - `.\.venv\Scripts\activate`
  - `pip install -r requirements.txt`

## Train / Refresh Model
- If you need to (re)build `model.pkl`:
  - `python train_model.py`

## Run API Service
- From repo root with venv active (bind to localhost only):
  - `uvicorn api:app --host localhost --port 8000`
- Health: `curl http://localhost:8000/health`
- Prediction example:
  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{
      \"operation_type\": \"Appendectomy\",
      \"tool_name\": \"Scalpel\",
      \"surgery_duration_min\": 90,
      \"complexity_score\": 3,
      \"surgeon_experience_years\": 10
    }"
  ```

## Run Streamlit UI
- From repo root with venv active (bind to localhost only):
  - `streamlit run app.py --server.headless true --server.port 8501 --server.address localhost`
- Open http://localhost:8501

## Run Everything with Docker Compose
- Prereqs: Docker Desktop running.
- Build and start the stack (FastAPI + Streamlit + Ollama):
  - `docker compose up --build`
- Pull the Llama model once (run while compose is up):
  - `docker compose exec ollama ollama pull llama3.2`
- Services:
  - API (port 8000): http://localhost:8000/health
  - Streamlit UI (port 8501): http://localhost:8501
  - Ollama API (port 11434, for testing): http://localhost:11434
- Stop and clean up:
  - `docker compose down`

## Llama Connectivity Troubleshooting
- If you run Streamlit locally (not in Docker) and see errors about `ollama` not resolving, ensure `OLLAMA_URL` is not set to `http://ollama:11434/...`; it should be `http://localhost:11434/api/generate` when running Ollama on your host.
- Connection refused means Ollama is not running or the model is missing. Start Ollama and pull the model:
  - Host: `ollama serve` (in one terminal), then `ollama pull llama3.2` (in another).
  - Docker: `docker compose up` (starts Ollama), then `docker compose exec ollama ollama pull llama3.2`.
- When using Docker Compose, the Streamlit and API services wait for Ollama to be healthy before starting.

## Stop Services
- Ctrl+C in each terminal, or on PowerShell: `Get-Process uvicorn*, streamlit | Stop-Process`
