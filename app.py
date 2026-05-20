import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


@st.cache_resource
def setup_rag():
    """Initialize and cache the retriever, LLM, and prompt template."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_multi": 0.7
        }
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant which answers the question. Answer the question based on the context given. If any question is out of the context reply with couldn't find the answer"),
        ("human", "Context: {context}\nQuestion: {question}")
    ])
    return retriever, llm, prompt


def get_answer(query: str) -> str:
    """Retrieve relevant context and generate an answer for the given query."""
    retriever, llm, prompt = setup_rag()
    docs = retriever.invoke(query)
    context = "".join([doc.page_content for doc in docs])
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
    response = llm.invoke(final_prompt)
    return response.content


def render_chat_ui():
    """Render the Streamlit chat interface."""
    st.title("📑 RAG Document Assistant")
    st.caption("Ask questions about your document!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    query = st.chat_input("Ask a question about the document...")

    if query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                answer = get_answer(query)
            st.write(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


if __name__ == "__main__":
    load_dotenv()
    render_chat_ui()