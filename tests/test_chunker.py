from rag.chunker import chunk_pages

def test_chunks_respect_size_and_keep_page():
    pages = [{"page": 3, "text": "word " * 1000}]
    chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=150)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 800 for c in chunks)
    assert all(c["page"] == 3 for c in chunks)