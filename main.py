from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.ingest import ingest_pdf
from app.retriever import retrieve
from app.prompt_builder import build_prompts
from app.llm_client import call_llm
from app.guardrails import apply_guardrails
from app.evaluator import rouge_score, llm_judge
import io

app = FastAPI(title="SmartDoc AI")

class QueryRequest(BaseModel):
    question: str
    collection: str
    provider: str = "groq"
    prompt_type: str = "cot"
    reference_answer: str = ""

@app.get("/")
def root():
    return {"status": "SmartDoc AI is running"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    content = await file.read()
    n = ingest_pdf(io.BytesIO(content), file.filename.replace(".pdf", ""))
    return {"collection": file.filename.replace(".pdf", ""), "chunks": n}

@app.post("/query")
async def query(req: QueryRequest):
    chunks = retrieve(req.question, req.collection)
    prompts = build_prompts(chunks, req.question)
    prompt = prompts.get(req.prompt_type, prompts["cot"])
    llm_result = call_llm(prompt, provider=req.provider)
    safe = apply_guardrails(llm_result["text"], chunks)
    scores = llm_judge(req.question, safe["answer"], chunks)
    if req.reference_answer:
        scores["rouge_l"] = rouge_score(safe["answer"], req.reference_answer)
    return {
        "answer": safe["answer"],
        "chunks": safe["chunks"],
        "flagged": safe["flagged"],
        "scores": scores,
        "meta": {
            "tokens": llm_result["tokens"],
            "latency": llm_result["latency"],
            "provider": req.provider
        }
    }