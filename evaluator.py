from rouge_score import rouge_scorer
from app.llm_client import call_llm

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_score(prediction: str, reference: str) -> float:
    return round(scorer.score(reference, prediction)["rougeL"].fmeasure, 3)

def llm_judge(question: str, answer: str, context: list[str]) -> dict:
    prompt = f"""Rate this answer on a scale of 1–5 for:
1. Faithfulness (does it match the context?)
2. Relevance (does it answer the question?)
Context: {' '.join(context[:2])}
Question: {question}
Answer: {answer}
Respond in this exact format:
Faithfulness: X
Relevance: X
Reason: one sentence"""
    result = call_llm(prompt, provider="groq")
    lines = result["text"].strip().split("\n")
    scores = {}
    for line in lines:
        if "Faithfulness:" in line:
            scores["faithfulness"] = float(line.split(":")[1].strip())
        elif "Relevance:" in line:
            scores["relevance"] = float(line.split(":")[1].strip())
        elif "Reason:" in line:
            scores["reason"] = line.split(":", 1)[1].strip()
    return scores