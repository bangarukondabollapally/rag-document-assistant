"""The original Streamlit interface, kept as a simple alternative to the React workspace."""

import streamlit as st

from document_store import DocumentStore, UploadRejected
from rag_core import SAMPLE_DOCUMENT, RagPipeline


@st.cache_resource
def get_store() -> DocumentStore:
    store = DocumentStore()
    store.load()
    return store


@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline()


def index_sample(store: DocumentStore) -> None:
    """Indexing runs only because the button was pressed."""
    try:
        document_id = store.add_file(SAMPLE_DOCUMENT).id
    except UploadRejected as exc:
        if not exc.document_id:
            raise
        document_id = exc.document_id
    store.index_now(document_id)


def choose_documents(store: DocumentStore) -> list[str]:
    ready = [record for record in store.list() if record.status == "ready"]
    if not ready:
        st.info("No documents are indexed yet.")
        if st.button("Index the sample document"):
            with st.spinner("Indexing the sample document..."):
                index_sample(store)
            st.rerun()
        return []

    names = {record.id: record.name for record in ready}
    return st.multiselect(
        "Documents to search",
        options=list(names),
        default=list(names)[:1],
        format_func=names.get,
    )


def render_chat_ui():
    st.title("📑 RAG Document Assistant")
    st.caption("Ask questions about your documents!")

    store = get_store()
    selected = choose_documents(store)
    if not selected:
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Ask a question about the selected documents...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer = get_pipeline().answer(store.searchable(selected), query)
            st.write(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


render_chat_ui()
