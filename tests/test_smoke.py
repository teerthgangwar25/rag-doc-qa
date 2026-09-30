import pytest
from rag.loader import load_pdf
from rag.chunker import chunk_pages
from rag.embedder import embed_documents, embed_query
from rag.store import VectorStore
from rag.chain import answer_question

@pytest.mark.smoke
def test_end_to_end_answers_a_real_question():
    chunks = chunk_pages(load_pdf("sample.pdf"))
    store = VectorStore()
    store.add(chunks, embed_documents([c["text"] for c in chunks]))

    results = store.search(embed_query("What is this document about?"), k=4)
    answer = answer_question("What is this document about?", results)

    assert isinstance(answer, str)
    assert len(answer) > 0