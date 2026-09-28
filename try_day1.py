from rag.loader import load_pdf
from rag.chunker import chunk_pages

pages = load_pdf("sample.pdf")
chunks = chunk_pages(pages)

print(len(pages), "pages ->", len(chunks), "chunks")
for c in chunks[:3]:
    print(f"[page {c['page']}] {len(c['text'])} chars: {c['text'][:80]!r}")