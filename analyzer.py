import json
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

from prompts import SYSTEM_PROMPT

load_dotenv()

_configured = False
_model = None


def get_model():
    """Lazily configure and build the Gemini model.

    Doing this at call-time (rather than at import-time) means a missing
    or invalid API key surfaces as a clean, catchable error inside the
    Streamlit UI instead of crashing the whole app on import.
    """
    global _configured, _model

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file or environment."
        )

    if not _configured:
        genai.configure(api_key=api_key)
        _configured = True

    if _model is None:
        _model = genai.GenerativeModel("gemini-2.5-pro")

    return _model


def build_analysis_prompt(messages):
    conversation = ""

    for m in messages:
        conversation += f"""
Message ID: {m['id']}
Date: {m['date']}
Time: {m['time']}
Sender: {m['sender']}
Message:
{m['text']}

-----------------------------------
"""

    return f"{SYSTEM_PROMPT}\n\nConversation\n{conversation}"


def build_compare_prompt(reports):
    serialized_reports = json.dumps(reports, indent=2)

    return f"""
You are comparing multiple per-conversation audit reports generated for Edoofa
counselors. Identify patterns that repeat ACROSS the reports below — do not
just restate any single report.

Return ONLY valid JSON in this exact format (fill in the values, do not
include the input reports in your output):

{{
  "common_strengths": [],
  "common_weaknesses": [],
  "recurring_parent_concerns": [],
  "recurring_student_concerns": [],
  "repeated_broken_promises": [],
  "repeated_compliance_risks": [],
  "training_priorities": [],
  "overall_organization_score": 0
}}

Reports to compare

{serialized_reports}
"""


def parse_json_response(response_text):
    # Gemini sometimes wraps JSON in ```json ... ``` fences even when asked
    # not to; strip those before parsing so we don't spuriously fail.
    text = response_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse JSON from Gemini response.",
            "raw_response": response_text,
        }


def _call_model(prompt, model=None, retries=2, backoff_seconds=2):
    """Call Gemini with basic retry handling and a clean error return
    instead of an uncaught exception blowing up the Streamlit app."""
    try:
        active_model = model or get_model()
    except RuntimeError as exc:
        return {"error": str(exc)}

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = active_model.generate_content(prompt)
            return parse_json_response(response.text)
        except Exception as exc:  # network errors, rate limits, etc.
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))

    return {
        "error": "Failed to get a response from Gemini.",
        "details": str(last_error),
    }


def analyze(messages, model=None):
    prompt = build_analysis_prompt(messages)
    return _call_model(prompt, model=model)


def compare_chats(reports, model=None):
    prompt = build_compare_prompt(reports)
    return _call_model(prompt, model=model)