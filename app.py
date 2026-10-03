import streamlit as st
from rag.loader import load_pdf
from rag.chunker import chunk_pages
from rag.embedder import embed_documents, embed_query
from rag.store import VectorStore
from rag.chain import answer_question

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A --V2")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    already_indexed = st.session_state.get("filename") == uploaded.name

    if not already_indexed:
        with st.spinner("Reading and indexing your PDF..."):
            try:
                pages = load_pdf(uploaded)
                chunks = chunk_pages(pages)
                vectors = embed_documents([c["text"] for c in chunks])
                store = VectorStore()
                store.add(chunks, vectors)

                st.session_state["store"] = store
                st.session_state["filename"] = uploaded.name
                st.session_state["chat"] = []
            except ValueError as e:
                st.error(str(e))
                st.stop()

    n_pages = max(c["page"] for c in st.session_state["store"].chunks)
    n_chunks = len(st.session_state["store"].chunks)
    st.success(f"Indexed **{uploaded.name}** — {n_pages} pages, {n_chunks} chunks")

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.caption(f"[page {s['page']}] score={s['score']:.2f}")
                        st.text(s["text"][:300])

    question = st.chat_input("Ask a question about your PDF")

    if question:
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                results = st.session_state["store"].search(embed_query(question), k=4)
                answer = answer_question(question, results)
                st.write(answer)
                with st.expander("Sources"):
                    for s in results:
                        st.caption(f"[page {s['page']}] score={s['score']:.2f}")
                        st.text(s["text"][:300])

        st.session_state["chat"].append(
            {"role": "assistant", "content": answer, "sources": results}
        )
else:
    st.info("Upload a PDF to get started.")