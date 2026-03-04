import json
import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi"   


def generate_structured_note(segment_text: str) -> dict:
    """
    Generates structured notes using Ollama HTTP API.
    Returns validated JSON dictionary.
    """

    prompt = f"""
You are a strict JSON generator.

TASK:
Convert the lecture segment into structured notes.

RULES:
- Respond ONLY with valid JSON.
- Do NOT add explanations.
- Do NOT add text before or after JSON.
- Use double quotes for all keys and strings.
- If unsure, return best possible structured summary.

FORMAT:

{{
  "title": "Short descriptive title (max 10 words)",
  "summary": "2-3 sentence concise summary",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "concepts": ["Concept 1", "Concept 2"]
}}

Return EXACTLY one JSON object.
Do not repeat.

LECTURE SEGMENT:
{segment_text}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0,
        "stream": False,
        "options": {
    "num_predict": 300
}
}

    try:
        response = requests.post(OLLAMA_URL, json=payload)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama connection failed: {e}")

    if response.status_code != 200:
        raise RuntimeError(f"Ollama API error: {response.text}")

    result = response.json()["response"].strip()

# Find JSON block
    start = result.find("{")
    end = result.rfind("}") + 1

    if start == -1 or end == -1:
        raise ValueError("Model did not return JSON.")

    json_str = result[start:end]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        print("RAW MODEL OUTPUT:", result)
        raise ValueError("Model returned malformed JSON.")

    return parsed