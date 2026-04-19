import re
from pydantic import BaseModel, validator

PII_PATTERNS = [
    r"\b\d{10,12}\b",                          # phone numbers
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-z]+", # emails
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"  # card numbers
]
TOXIC_KEYWORDS = ["kill", "hate", "violence", "abuse"]

class LLMResponse(BaseModel):
    answer: str
    source_chunks: list[str]

    @validator("answer")
    def must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v

def redact_pii(text: str) -> str:
    for pattern in PII_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text

def check_toxicity(text: str) -> bool:
    return any(word in text.lower() for word in TOXIC_KEYWORDS)

def apply_guardrails(answer: str, chunks: list[str]) -> dict:
    answer = redact_pii(answer)
    flagged = check_toxicity(answer)
    try:
        validated = LLMResponse(answer=answer, source_chunks=chunks)
        return {"answer": validated.answer, "chunks": chunks, "pii_redacted": True, "flagged": flagged}
    except Exception as e:
        return {"answer": "Unable to generate a safe response.", "chunks": [], "pii_redacted": True, "flagged": True}