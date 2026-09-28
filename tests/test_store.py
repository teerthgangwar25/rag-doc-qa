from rag.store import VectorStore

def make_store():
    store = VectorStore(dim=3)
    chunks = [
        {"text": "A", "page": 1},
        {"text": "B", "page": 2},
        {"text": "C", "page": 3},
    ]
    store.add(chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return store

def test_search_returns_closest_chunk():
    results = make_store().search([0, 0.9, 0.1], k=2)
    assert results[0]["text"] == "B"
    assert len(results) == 2

def test_search_with_k_larger_than_index():
    results = make_store().search([1, 0, 0], k=5)
    assert len(results) == 3