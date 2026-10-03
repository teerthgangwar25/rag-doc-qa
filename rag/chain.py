import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a document Q&A assistant.
Answer ONLY using the context below. Do not use outside knowledge.
If the answer is not in the context, say: "I couldn't find that in the document."
Cite the page number(s) you used, like (page 3)."""

def _client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to .env")
    return Groq(api_key=key.strip())

def build_context(chunks):
    return "\n\n".join(f"[page {c['page']}]\n{c['text']}" for c in chunks)

def answer_question(question, chunks):
    context = build_context(chunks)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    resp = _client().chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content