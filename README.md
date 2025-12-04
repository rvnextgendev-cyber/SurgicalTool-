# Local LLM Demo

Simple FastAPI + Streamlit demo for surgical tool usage prediction.

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

## Stop Services
- Ctrl+C in each terminal, or on PowerShell: `Get-Process uvicorn*, streamlit | Stop-Process`
