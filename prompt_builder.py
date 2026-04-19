from jinja2 import Template

ZERO_SHOT = Template("""You are a helpful assistant. Answer the question using only the context below.
Context:
{{ context }}
Question: {{ question }}
Answer:""")

FEW_SHOT = Template("""You are a helpful assistant. Here are examples of good answers:
Q: What is the main topic? A: The document discusses...
Q: Who is the author? A: The author is...
Now answer:
Context: {{ context }}
Question: {{ question }}
Answer:""")

COT = Template("""You are a helpful assistant. Think step by step before answering.
Context: {{ context }}
Question: {{ question }}
Let's think step by step:
1. Identify the relevant section.
2. Extract the key fact.
3. Formulate the answer.
Answer:""")

def build_prompts(context: str, question: str) -> dict:
    ctx = "\n".join(context)
    return {
        "zero_shot": ZERO_SHOT.render(context=ctx, question=question),
        "few_shot":  FEW_SHOT.render(context=ctx, question=question),
        "cot":       COT.render(context=ctx, question=question),
    }