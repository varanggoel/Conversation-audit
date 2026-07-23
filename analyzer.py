import json
import os
import re
import time
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency guard
    genai = None

from prompts import SYSTEM_PROMPT

load_dotenv()

_configured = False
_model = None


def get_model():
    """Lazily configure and build the Gemini model when available."""
    global _configured, _model

    if genai is None:
        raise RuntimeError("google-generativeai is not installed")

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
    """Call Gemini with basic retry handling and a clean error return."""
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


def _normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def _message_ids(messages):
    return [m["id"] for m in messages]


def _build_framework():
    return [
        {
            "name": "Professionalism",
            "measure": "Tone, courtesy, and clarity of counsel.",
            "why": "Edoofa's brand depends on counselors sounding reliable and respectful.",
        },
        {
            "name": "Responsiveness",
            "measure": "Whether questions and concerns are answered promptly.",
            "why": "Parents and students need timely guidance during critical decision moments.",
        },
        {
            "name": "Information Accuracy",
            "measure": "Consistency of facts about fees, admissions, and next steps.",
            "why": "Inaccurate information creates trust and compliance risk.",
        },
        {
            "name": "Broken Promises",
            "measure": "Promises made and later not followed through.",
            "why": "Broken commitments erode confidence quickly in a service-led business.",
        },
        {
            "name": "Transparency",
            "measure": "Clarity about what is known, unknown, and what will happen next.",
            "why": "Transparent communication protects trust with families.",
        },
        {
            "name": "Empathy",
            "measure": "Whether the conversation reflects care for the student's situation.",
            "why": "Students and families need emotional reassurance as well as guidance.",
        },
        {
            "name": "Sales Pressure",
            "measure": "Urgency or coercive language that may push families too quickly.",
            "why": "Edoofa should build trust rather than pressure families into rushed decisions.",
        },
        {
            "name": "Personalization",
            "measure": "Use of the student's background or specific concerns in the conversation.",
            "why": "Personalized counseling feels more credible and helpful.",
        },
        {
            "name": "Trust Building",
            "measure": "Signals of reliability, ownership, and follow-through.",
            "why": "High-trust conversations increase conversion and retention.",
        },
        {
            "name": "Concern Resolution",
            "measure": "Whether the counselor resolves or clearly addresses the concern raised.",
            "why": "Unresolved concerns become escalations and churn risk.",
        },
        {
            "name": "Follow-up Quality",
            "measure": "Quality of the next steps and follow-up commitments.",
            "why": "Good follow-up prevents confusion and lost opportunities.",
        },
        {
            "name": "Tone Changes",
            "measure": "Shifts from calm to anxious, defensive, or dismissive language.",
            "why": "Tone shifts can signal poor handling of sensitive conversations.",
        },
        {
            "name": "Ethical Counseling",
            "measure": "Whether the counselor stays within an ethical, supportive standard.",
            "why": "Ethical conversations protect the student and the organization.",
        },
        {
            "name": "Compliance Risks",
            "measure": "Potential issues around privacy, consent, or policy-sensitive statements.",
            "why": "Compliance problems can create legal and brand risk.",
        },
        {
            "name": "Conversation Flow",
            "measure": "How smoothly the discussion moves from question to answer to next step.",
            "why": "A coherent flow makes counseling easier to follow and trust.",
        },
    ]


def _default_parameters():
    return {
        "professionalism": {"urgency_penalty": 10, "uncertainty_penalty": 5, "courtesy_bonus": 2},
        "responsiveness": {"question_penalty": 12, "max_penalty": 24},
        "information_accuracy": {"uncertainty_penalty": 10, "detail_penalty": 2},
        "broken_promises": {"promise_penalty": 15},
        "transparency": {"uncertainty_penalty": 10},
        "empathy": {"empathy_bonus": 10, "default_score": 80},
        "sales_pressure": {"urgency_penalty": 20},
        "personalization": {"personalization_bonus": 20, "default_score": 70},
        "trust_building": {"followup_bonus": 10, "uncertainty_penalty": 10},
        "concern_resolution": {"unresolved_penalty": 15},
        "follow_up_quality": {"followup_bonus": 10},
        "tone_changes": {"negative_penalty": 15},
        "ethical_counseling": {"urgency_penalty": 10},
        "compliance_risks": {"sensitive_penalty": 5},
        "conversation_flow": {"missing_counselor_penalty": 10},
    }


def _heuristic_analyze(messages):
    parameters = _default_parameters()

    if not messages:
        return {
            "overall_score": 0,
            "risk": "Low",
            "findings": [],
            "category_scores": {category["name"]: 0 for category in _build_framework()},
            "framework": _build_framework(),
            "strengths": [],
            "coaching_priorities": ["Upload at least one conversation to start the audit."],
            "parameters": parameters,
        }

    normalized_messages = [
        {**message, "normalized_text": _normalize_text(message.get("text", ""))}
        for message in messages
    ]
    all_text = " ".join(message["normalized_text"] for message in normalized_messages)

    scores = {}
    findings = []

    counselor_messages = [m for m in normalized_messages if "counselor" in m["sender"].lower()]
    parent_or_student = [m for m in normalized_messages if "parent" in m["sender"].lower() or "student" in m["sender"].lower()]

    empathy_terms = ["understand", "sorry", "appreciate", "concern", "happy", "support", "help"]
    pressure_terms = ["today", "urgent", "limited", "hurry", "now", "don't miss", "must", "immediately", "only today"]
    uncertainty_terms = ["not sure", "maybe", "probably", "i think", "will check", "let me confirm"]
    promise_terms = ["will", "confirm", "send", "call", "update", "share", "follow up"]
    follow_up_terms = ["follow up", "next step", "will send", "will call", "will share", "update you"]
    personalization_terms = ["your", "your child", "your daughter", "your son", "your profile", "your goal"]

    professional_score = 85
    if any(term in all_text for term in pressure_terms):
        professional_score -= parameters["professionalism"]["urgency_penalty"]
    if any(term in all_text for term in uncertainty_terms):
        professional_score -= parameters["professionalism"]["uncertainty_penalty"]
    if any(term in all_text for term in ["please", "thank you", "appreciate"]):
        professional_score += parameters["professionalism"]["courtesy_bonus"]

    responsiveness_score = 80
    unanswered_questions = []
    for idx, message in enumerate(normalized_messages):
        text = message["normalized_text"]
        if "?" in text and not any(keyword in text for keyword in ["please", "thanks"]):
            if idx == len(normalized_messages) - 1:
                unanswered_questions.append(message["id"])
            else:
                replied = False
                for later in normalized_messages[idx + 1:]:
                    if "counselor" in later["sender"].lower() and later["id"] != message["id"]:
                        replied = True
                        break
                if not replied:
                    unanswered_questions.append(message["id"])
    if unanswered_questions:
        responsiveness_score -= min(
            parameters["responsiveness"]["max_penalty"],
            parameters["responsiveness"]["question_penalty"] * len(unanswered_questions),
        )
        findings.append({
            "category": "Responsiveness",
            "severity": "High" if len(unanswered_questions) > 1 else "Medium",
            "summary": "The conversation contains questions that were not clearly answered.",
            "evidence": unanswered_questions,
            "why": "Families often raise urgent concerns that need a clear response before they can move forward.",
            "recommendation": "Answer every raised question explicitly and note the next step or owner.",
        })

    accuracy_score = 85
    if any(term in all_text for term in uncertainty_terms):
        accuracy_score -= parameters["information_accuracy"]["uncertainty_penalty"]
    if any(term in all_text for term in ["fee", "scholarship", "deadline", "admission"]):
        accuracy_score -= parameters["information_accuracy"]["detail_penalty"]
    if accuracy_score < 80:
        findings.append({
            "category": "Information Accuracy",
            "severity": "Medium",
            "summary": "The conversation contains uncertain or potentially inconsistent statements.",
            "evidence": _message_ids(normalized_messages),
            "why": "Confusion around admissions facts or costs harms trust and creates compliance exposure.",
            "recommendation": "Use documented facts and verify details before sharing them with families.",
        })

    broken_promise_score = 90
    promise_messages = [m["id"] for m in normalized_messages if any(term in m["normalized_text"] for term in promise_terms)]
    if promise_messages and len(counselor_messages) > 0:
        broken_promise_score -= parameters["broken_promises"]["promise_penalty"]
        findings.append({
            "category": "Broken Promises",
            "severity": "Medium",
            "summary": "The counselor uses commitment language that is not reinforced by a clear follow-up.",
            "evidence": promise_messages,
            "why": "Promising a callback, document, or update without follow-through damages credibility.",
            "recommendation": "Only promise what can be delivered and confirm the delivery timeline.",
        })

    transparency_score = 85
    if any(term in all_text for term in uncertainty_terms):
        transparency_score -= parameters["transparency"]["uncertainty_penalty"]
    if any(term in all_text for term in ["not sure", "will check"]):
        findings.append({
            "category": "Transparency",
            "severity": "Medium",
            "summary": "The counselor uses hedge language instead of confirming the next step.",
            "evidence": [m["id"] for m in normalized_messages if any(term in m["normalized_text"] for term in uncertainty_terms)],
            "why": "Families need to know whether the counselor can answer now or must follow up later.",
            "recommendation": "State what is confirmed, what is pending, and when the family will hear back.",
        })

    empathy_score = parameters["empathy"]["default_score"]
    if any(term in all_text for term in empathy_terms):
        empathy_score += parameters["empathy"]["empathy_bonus"]
    else:
        findings.append({
            "category": "Empathy",
            "severity": "Low",
            "summary": "The counselor does not make explicit empathetic signals.",
            "evidence": _message_ids(normalized_messages),
            "why": "Empathy reduces stress and helps families feel supported.",
            "recommendation": "Add a short acknowledgement of the family's concern before giving the next step.",
        })

    sales_pressure_score = 90
    if any(term in all_text for term in pressure_terms):
        sales_pressure_score -= parameters["sales_pressure"]["urgency_penalty"]
        findings.append({
            "category": "Sales Pressure",
            "severity": "Medium",
            "summary": "The conversation uses urgency-based language that could feel pressuring.",
            "evidence": [m["id"] for m in normalized_messages if any(term in m["normalized_text"] for term in pressure_terms)],
            "why": "Pressure can undermine trust and make families feel rushed.",
            "recommendation": "Replace urgency cues with calm guidance and clear choices.",
        })

    personalization_score = parameters["personalization"]["default_score"]
    if any(term in all_text for term in personalization_terms):
        personalization_score += parameters["personalization"]["personalization_bonus"]
    if not parent_or_student:
        personalization_score -= 10
    if personalization_score < 80:
        findings.append({
            "category": "Personalization",
            "severity": "Low",
            "summary": "The counselor does not clearly tailor the conversation to the student's stated context.",
            "evidence": _message_ids(normalized_messages),
            "why": "Specific references make the guidance feel relevant and credible.",
            "recommendation": "Reference the student's situation, goals, or prior concerns in follow-up messages.",
        })

    trust_score = 80
    if any(term in all_text for term in follow_up_terms):
        trust_score += parameters["trust_building"]["followup_bonus"]
    if any(term in all_text for term in uncertainty_terms):
        trust_score -= parameters["trust_building"]["uncertainty_penalty"]
    if trust_score < 80:
        findings.append({
            "category": "Trust Building",
            "severity": "Medium",
            "summary": "The conversation could do more to reinforce reliability and ownership.",
            "evidence": _message_ids(normalized_messages),
            "why": "Consistent ownership is a cornerstone of high-quality admissions counseling.",
            "recommendation": "Close each interaction with a clear owner, timing, and next step.",
        })

    concern_resolution_score = 80
    if unanswered_questions:
        concern_resolution_score -= parameters["concern_resolution"]["unresolved_penalty"]
    if concern_resolution_score < 80:
        findings.append({
            "category": "Concern Resolution",
            "severity": "Medium",
            "summary": "Some concerns raised by the family do not appear fully resolved.",
            "evidence": unanswered_questions or _message_ids(normalized_messages),
            "why": "Unresolved concerns can quickly become escalations or dropped applications.",
            "recommendation": "Restate the concern and confirm the answer before moving on.",
        })

    follow_up_score = 80
    if any(term in all_text for term in follow_up_terms):
        follow_up_score += parameters["follow_up_quality"]["followup_bonus"]
    else:
        findings.append({
            "category": "Follow-up Quality",
            "severity": "Low",
            "summary": "There are few explicit follow-up or next-step cues in the conversation.",
            "evidence": _message_ids(normalized_messages),
            "why": "Strong follow-up lowers confusion and keeps the student moving forward.",
            "recommendation": "Confirm the next action, owner, and timing for every conversation.",
        })

    tone_score = 85
    negative_tone_terms = ["frustrated", "angry", "upset", "confused", "worried", "concerned", "problem", "issue"]
    if any(term in all_text for term in negative_tone_terms):
        tone_score -= parameters["tone_changes"]["negative_penalty"]
    if tone_score < 80:
        findings.append({
            "category": "Tone Changes",
            "severity": "Medium",
            "summary": "The conversation carries visible tension or frustration markers.",
            "evidence": [m["id"] for m in normalized_messages if any(term in m["normalized_text"] for term in negative_tone_terms)],
            "why": "Tone drift can quickly undermine confidence and rapport.",
            "recommendation": "Acknowledge the emotion first and then re-anchor the conversation on next steps.",
        })

    ethical_score = 90
    if any(term in all_text for term in pressure_terms):
        ethical_score -= parameters["ethical_counseling"]["urgency_penalty"]
    if ethical_score < 90:
        findings.append({
            "category": "Ethical Counseling",
            "severity": "Low",
            "summary": "The conversation uses forceful language that should be softened.",
            "evidence": [m["id"] for m in normalized_messages if any(term in m["normalized_text"] for term in pressure_terms)],
            "why": "Ethical counseling should protect autonomy rather than create pressure.",
            "recommendation": "Offer options and explain trade-offs instead of pushing urgency.",
        })

    compliance_score = 88
    if any(term in all_text for term in ["payment", "fee", "scholarship", "document", "id"]):
        compliance_score -= parameters["compliance_risks"]["sensitive_penalty"]
    if compliance_score < 90:
        findings.append({
            "category": "Compliance Risks",
            "severity": "Low",
            "summary": "The chat discusses sensitive information that should be handled with clear policy guardrails.",
            "evidence": [m["id"] for m in normalized_messages if any(term in m["normalized_text"] for term in ["payment", "fee", "scholarship", "document", "id"])],
            "why": "Sensitive student and family information requires careful handling.",
            "recommendation": "Reference the approved process and avoid sharing sensitive information outside the approved channel.",
        })

    flow_score = 85
    if len(normalized_messages) > 1 and not counselor_messages:
        flow_score -= parameters["conversation_flow"]["missing_counselor_penalty"]
    scores.update(
        {
            "Professionalism": professional_score,
            "Responsiveness": responsiveness_score,
            "Information Accuracy": accuracy_score,
            "Broken Promises": broken_promise_score,
            "Transparency": transparency_score,
            "Empathy": empathy_score,
            "Sales Pressure": sales_pressure_score,
            "Personalization": personalization_score,
            "Trust Building": trust_score,
            "Concern Resolution": concern_resolution_score,
            "Follow-up Quality": follow_up_score,
            "Tone Changes": tone_score,
            "Ethical Counseling": ethical_score,
            "Compliance Risks": compliance_score,
            "Conversation Flow": flow_score,
        }
    )

    overall_score = int(round(sum(scores.values()) / len(scores)))
    if overall_score > 90:
        risk = "Low"
    elif overall_score > 75:
        risk = "Medium"
    else:
        risk = "High"

    if not findings:
        findings.append({
            "category": "Conversation Flow",
            "severity": "Low",
            "summary": "No major issues were detected in the conversation.",
            "evidence": _message_ids(normalized_messages),
            "why": "The conversation remained coherent and did not show obvious quality risks.",
            "recommendation": "Keep the current style and reinforce it with consistent follow-up habits.",
        })

    strengths = []
    if any(term in all_text for term in ["thank you", "appreciate", "help"]):
        strengths.append("The counselor uses courteous language that supports a professional tone.")
    if any(term in all_text for term in follow_up_terms):
        strengths.append("The counselor provides clear next-step language.")
    if any(term in all_text for term in empathy_terms):
        strengths.append("The conversation acknowledges the family's concern with empathy.")

    coaching_priorities = [
        "Address every student or parent question explicitly.",
        "Use calm, non-pressuring language.",
        "Confirm the next step and owner for every conversation.",
    ]

    return {
        "overall_score": overall_score,
        "risk": risk,
        "findings": findings,
        "category_scores": scores,
        "framework": _build_framework(),
        "strengths": strengths or ["The counselor maintained a generally coherent exchange."],
        "coaching_priorities": coaching_priorities,
        "parameters": parameters,
    }


def analyze(messages, model=None):
    prompt = build_analysis_prompt(messages)
    if model is not None:
        return _call_model(prompt, model=model)

    try:
        result = _call_model(prompt, model=None)
    except Exception:
        result = {"error": "Analyzer unavailable"}

    if isinstance(result, dict) and "error" not in result:
        return result
    return _heuristic_analyze(messages)


def compare_chats(reports, model=None):
    if not reports:
        return {"common_strengths": [], "common_weaknesses": [], "recurring_parent_concerns": [], "recurring_student_concerns": [], "repeated_broken_promises": [], "repeated_compliance_risks": [], "training_priorities": [], "overall_organization_score": 0}

    if model is not None:
        prompt = build_compare_prompt(reports)
        return _call_model(prompt, model=model)

    common_strengths = []
    common_weaknesses = []
    recurring_parent_concerns = []
    recurring_student_concerns = []
    repeated_broken_promises = []
    repeated_compliance_risks = []
    training_priorities = []

    for report in reports:
        for finding in report.get("findings", []):
            common_weaknesses.append(finding["category"])
        for priority in report.get("coaching_priorities", []):
            training_priorities.append(priority)

    if common_weaknesses:
        common_weaknesses = sorted(set(common_weaknesses))
    if training_priorities:
        training_priorities = list(dict.fromkeys(training_priorities))

    return {
        "common_strengths": common_strengths,
        "common_weaknesses": common_weaknesses,
        "recurring_parent_concerns": recurring_parent_concerns,
        "recurring_student_concerns": recurring_student_concerns,
        "repeated_broken_promises": repeated_broken_promises,
        "repeated_compliance_risks": repeated_compliance_risks,
        "training_priorities": training_priorities,
        "overall_organization_score": int(round(sum(report.get("overall_score", 0) for report in reports) / max(1, len(reports)))),
    }