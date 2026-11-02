import streamlit as st

from rag_core import DocumentError, IndexNotAvailable, open_saved_pipeline, rebuild_index


# Failures aren't cached, so this tries again on the next run.
@st.cache_resource
def setup_rag():
    return open_saved_pipeline()


def index_is_ready() -> bool:
    try:
        setup_rag()
        return True
    except DocumentError as exc:
        st.error(str(exc))
    except IndexNotAvailable as exc:
        st.warning(f"No index available. {exc}")
        if st.button("Build index"):
            with st.spinner("Building the index. This can take a few minutes on a CPU."):
                rebuild_index()
            st.rerun()
    return False


def get_answer(query: str) -> str:
    return setup_rag().answer(query)


def render_chat_ui():
    st.title("📑 RAG Document Assistant")
    st.caption("Ask questions about your document!")

    if not index_is_ready():
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Ask a question about the document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                answer = get_answer(query)
            st.write(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


render_chat_ui()