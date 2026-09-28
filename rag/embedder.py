import os
import cohere
from dotenv import load_dotenv

load_dotenv()

MODEL = "embed-v4.0"
DIM = 1024
BATCH = 96

def _client():
    key = os.getenv("COHERE_API_KEY")
    if not key:
        raise RuntimeError("COHERE_API_KEY not set. Add it to .env")
    return cohere.ClientV2(api_key=key)

def _embed(texts, input_type):
    co = _client()
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        res = co.embed(
            texts=batch,
            model=MODEL,
            input_type=input_type,
            output_dimension=DIM,
            embedding_types=["float"],
        )
        vectors.extend(res.embeddings.float)
    return vectors

def embed_documents(texts):
    return _embed(texts, "search_document")

def embed_query(text):
    return _embed([text], "search_query")[0]