# RAG Document Q&A

Upload a PDF, ask it questions, get answers grounded in what's actually on the page, not what an LLM assumes should be there.

**Live app:** [https://rag-doc-app-tg.streamlit.app/](#) · **Repo:** [github.com/teerthgangwar25/rag-doc-qa](https://github.com/teerthgangwar25/rag-doc-qa)

---

## What it does

You upload a PDF. It gets split into chunks, embedded, and dropped into a FAISS index. When you ask a question, the app finds the four most relevant chunks and hands them to an LLM with one instruction: answer from this context only, and say so if the answer isn't here. The model cites the page number it pulled from, and you can expand a "Sources" panel under every answer to check it isn't making things up.

That last part matters more than it sounds. Most RAG demos skip the "I don't know" case entirely and just let the model guess confidently from its training data. This one doesn't.

## How it's built

```
PDF upload
    │
    ▼
Extract text per page (pypdf)
    │
    ▼
Split into ~800-char chunks, 150-char overlap (LangChain's recursive splitter)
    │
    ▼
Embed each chunk (Cohere embed-v4.0, input_type=search_document)
    │
    ▼
Store in FAISS (IndexFlatIP, cosine similarity via L2 normalization)
    │
    ▼
┌─── question comes in ───┐
│  Embed the question      │
│  (input_type=search_query)│
│         │                │
│         ▼                │
│  Search top-4 chunks      │
│         │                │
│         ▼                │
│  Groq (openai/gpt-oss-120b)│
│  answers from context only│
└───────────────────────────┘
```

## Design decisions worth explaining

A few choices here aren't the default tutorial settings, and each one came from a reason, not a guess.

**Chunk size is 800 characters with 150 overlap.** Too small and you lose context mid-thought. Too large and retrieval gets fuzzy because one chunk tries to cover too much ground. 800 with overlap keeps most chunks inside a single idea while still catching answers that happen to sit right on a chunk boundary.

**Cohere's `search_document` and `search_query` input types are never mixed up.** Cohere optimizes embeddings differently depending on whether you're indexing a passage or searching with a question. Mix them up and nothing crashes, retrieval just quietly gets worse, which is the kind of bug that's easy to ship without noticing.

**The LLM is `openai/gpt-oss-120b` on Groq, not Llama.** The original plan called for Llama 3.3 70B. Turns out my Groq account doesn't have Llama models enabled at all, only GPT-OSS, Qwen, and a few others. Rather than wait on access, I checked what was actually available and used that instead. Same API shape, same result.

**It's hosted on Streamlit Community Cloud, not Hugging Face Spaces.** The original plan was HF Spaces. Partway through, I found out HF now requires a paid plan for Gradio and Docker SDKs, the only two that can run a Python backend. Static Spaces stay free, but a static site can't run FAISS or call an API. Streamlit Cloud does the same job for free and is built specifically for Streamlit apps.

**The grounding prompt explicitly tells the model to admit when it doesn't know.** Without that one line, GPT-OSS (like most LLMs) will guess a plausible-sounding answer from its own training data instead of saying the document doesn't cover it. Deleting that sentence and testing an out-of-scope question side by side is the fastest way to see why it's there.

## Stack

| Piece | Tool | Why |
|---|---|---|
| PDF parsing | pypdf | Extracts text with page numbers intact |
| Chunking | LangChain's `RecursiveCharacterTextSplitter` | Splits on paragraph/sentence boundaries before falling back to raw characters |
| Embeddings | Cohere `embed-v4.0` | Free trial tier, strong retrieval quality |
| Vector store | FAISS (`IndexFlatIP`) | Exact search, fast enough for a single document at this scale |
| LLM | Groq, `openai/gpt-oss-120b` | Free tier, ~320 tokens/sec |
| UI | Streamlit | Chat interface, session-state caching so it doesn't re-embed on every keystroke |
| CI | GitHub Actions | Unit tests plus one real end-to-end smoke test, every push |
| CD | Streamlit Community Cloud | Auto-redeploys `main` on every push, no manual step |

## CI/CD

Every push to `main` runs two tiers of tests in GitHub Actions:

- **Unit tests** (no API keys needed): chunking logic, FAISS search behavior, prompt construction. Fast, free, run on every push.
- **Smoke test** (needs real API keys, stored as GitHub secrets): loads an actual PDF, runs it through the full pipeline, and checks it returns a real answer. This is the one that catches "it works on my machine but the deployed version is broken" before it ships.

Deployment itself is handled by Streamlit Community Cloud, which watches the same repo and rebuilds automatically whenever `main` updates. GitHub Actions is the gate, Streamlit Cloud is what ships it, and neither needs a human to click anything.

## Running it locally

```bash
git clone https://github.com/teerthgangwar25/rag-doc-qa.git
cd rag-doc-qa
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
COHERE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Then:

```bash
streamlit run app.py
```

## Testing

```bash
python -m pytest -m "not smoke"   # fast, no keys required
python -m pytest -m smoke         # full pipeline, needs real API keys
```

## What I'd build next

- Chat history across questions, right now each question is answered independently
- Multiple PDFs in one session
- A small fixed set of test questions with known answers, to measure retrieval quality numerically instead of eyeballing it
- OCR fallback for scanned PDFs, which currently fail with a clear error instead of silently returning nothing

## License

MIT
