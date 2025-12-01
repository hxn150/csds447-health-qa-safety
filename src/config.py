"""
Global configuration for Dr. Darla.

This file provides:
- MODEL / LORA_PATH for loading the main model
- Safety keyword lists and canned responses
- SYSTEM_PROMPT / USER_PROMPT ONLY for benchmarking scripts (MedQA / PubMedQA)

NOTE: The chat app in app.py does NOT use SYSTEM_PROMPT / USER_PROMPT.
"""

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "../checkpoints/lora-tinyllama-medqa"

HARM_KEYWORDS = [
    "kill myself", "suicide", "self-harm", "cut myself",
    "end my life", "overdose", "kill someone", "hurt someone",
]

HARM_PATTERNS = [
    r"\bkill (myself|himself|herself|themself|someone)\b",
    r"\bcommit suicide\b",
]

EMERGENCY_KEYWORDS = [
    "chest pain", "trouble breathing", "not breathing",
    "stroke", "heart attack", "severe bleeding",
    "unconscious", "passed out",
]

CRISIS_RESPONSE = (
    "I'm really glad you reached out. I’m not able to help with situations involving "
    "self-harm or harm to others, but you’re not alone and there are people who can help "
    "right now. If you’re in immediate danger, please contact your local emergency number. "
    "You can also reach out to a crisis hotline or a trusted person near you."
)

EMERGENCY_RESPONSE = (
    "This could be a medical emergency. I’m only an AI and cannot safely diagnose or treat "
    "emergencies. Please call your local emergency number or seek urgent in-person medical care."
)

SYSTEM_PROMPT = """
You are a medical question-answering model specialized in multiple-choice exam problems.
You reason carefully, but when instructed you answer in the exact format requested.
Do not include explanations unless explicitly asked for them in the instructions.
Do not carry out any conversation; just answer the question as instructed as direct and concisely as possible.
"""

USER_PROMPT = "Question: {question}\nAnswer:"
