"""
Streamlit UI for surgical tool usage prediction with local Llama explanations.
Run the FastAPI service separately (uvicorn api:app --reload --port 8000) and this app will call it.
"""

import requests
import streamlit as st

from llama_client import call_local_llama

PREDICT_URL = "http://localhost:8000/predict"


def call_predict_api(payload: dict):
    try:
        res = requests.post(PREDICT_URL, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        return data.get("predicted_usage"), data.get("raw_prediction")
    except Exception as ex:  # noqa: BLE001
        st.error(f"Prediction API error: {ex}")
        return None, None


st.title("Surgical Tool Usage Prediction Demo")

st.markdown(
    """
This app demonstrates an **end-to-end AI/ML system**:

- A **trained ML regression model** served via FastAPI predicts tool usage.
- A **local Llama 3.2 model (Docker via Ollama)** explains the prediction in plain language.
"""
)

st.subheader("1) Enter surgery details")

operation_type = st.selectbox(
    "Operation type",
    ["Appendectomy", "C-Section", "Knee Replacement", "CABG", "Cholecystectomy"],
)

tool_name = st.selectbox(
    "Tool", ["Scalpel", "Forceps", "Retractor", "Suction", "Laparoscope"]
)

surgery_duration_min = st.slider("Surgery duration (minutes)", 30, 300, 90, 5)
complexity_score = st.slider("Complexity (1 = simple, 5 = very complex)", 1, 5, 3, 1)
surgeon_experience_years = st.slider(
    "Surgeon experience (years)",
    1,
    30,
    10,
    1,
)

if st.button("Predict tool usage"):
    payload = {
        "operation_type": operation_type,
        "tool_name": tool_name,
        "surgery_duration_min": surgery_duration_min,
        "complexity_score": complexity_score,
        "surgeon_experience_years": surgeon_experience_years,
    }

    predicted_usage, raw_prediction = call_predict_api(payload)

    if predicted_usage is not None:
        st.subheader("2) Predicted usage")
        st.success(
            f"Estimated usage for **{tool_name}** in this **{operation_type}**: "
            f"**{predicted_usage} times**"
        )

        if raw_prediction is not None:
            st.write(f"Raw model output (not rounded): `{raw_prediction:.2f}` uses")

        st.subheader("3) AI explanation (local Llama 3.2)")

        explain_prompt = f"""
You are a medical data assistant.

I have a model that predicts how many times a surgical tool will be used in an operation.

Given this case:

- Operation type: {operation_type}
- Tool: {tool_name}
- Surgery duration (minutes): {surgery_duration_min}
- Complexity (1-5): {complexity_score}
- Surgeon experience (years): {surgeon_experience_years}

The model predicts that the tool will be used approximately {predicted_usage} times.

In 4-6 simple sentences, explain **why** this might be reasonable,
based on the duration, complexity, and experience.
Do not mention that this is synthetic or a demo. Use simple language suitable for doctors and OR staff.
"""

        explanation = call_local_llama(explain_prompt)
        st.info(explanation)

        st.caption(
            "This is a demo using synthetic data. For real clinical use, you must train on "
            "real hospital data, validate with clinicians, and follow all regulations."
        )
