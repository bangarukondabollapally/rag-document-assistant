"""The original Streamlit interface, kept as a simple alternative to the React workspace."""

import streamlit as st

from document_store import DocumentStore, UploadRejected
from rag_core import RagPipeline


@st.cache_resource
def get_store() -> DocumentStore:
    store = DocumentStore()
    store.load()
    return store


@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline()


def add_document(store: DocumentStore) -> None:
    """Indexing runs only because the button was pressed."""
    file = st.file_uploader("Upload a PDF", type="pdf")
    if file is None or not st.button("Add and index"):
        return
    try:
        with st.spinner(f"Indexing {file.name}..."):
            record = store.index_now(store.add_upload(file, file.name, "application/pdf").id)
    except UploadRejected as exc:
        st.error(exc.message)
        return
    if record.status == "failed":
        # This interface has no retry, and a stored copy would block uploading it again.
        store.delete(record.id)
        st.error(record.error)
        return
    st.session_state.selected_documents = [record.id]


def choose_documents(store: DocumentStore) -> list[str]:
    ready = [record for record in store.list() if record.status == "ready"]
    if not ready:
        st.info("No documents are indexed yet. Upload a PDF to get started.")
        return []

    names = {record.id: record.name for record in ready}
    current = st.session_state.get("selected_documents", list(names)[:1])
    st.session_state.selected_documents = [document_id for document_id in current if document_id in names]
    return st.multiselect(
        "Documents to search",
        options=list(names),
        format_func=names.get,
        key="selected_documents",
    )


def render_chat_ui():
    st.title("📑 RAG Document Assistant")
    st.caption("Ask questions about your documents!")

    store = get_store()
    add_document(store)
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
