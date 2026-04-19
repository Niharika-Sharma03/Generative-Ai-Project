import os
import time
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt: str, provider: str = "groq") -> dict:
    start = time.time()
    try:
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        text = r.choices[0].message.content
        tokens = r.usage.total_tokens
        return {
            "text": text,
            "tokens": tokens,
            "latency": round(time.time() - start, 2),
            "provider": "groq"
        }
    except Exception as e:
        return {
            "text": f"Error: {e}",
            "tokens": 0,
            "latency": 0,
            "provider": "groq"
        }