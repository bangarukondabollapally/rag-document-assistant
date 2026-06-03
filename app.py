import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


@st.cache_resource
def setup_rag():
    loader = PyPDFLoader("DocumentLoaders/deeplearning.pdf")
    documents = loader.load()
    documents = documents[20:-10]  # skip cover/index pages like before

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant which answers the question. Answer the question based on the context given. If any question is out of the context reply with couldn't find the answer"),
        ("human", "Context: {context}\nQuestion: {question}")
    ])

    return retriever, llm, prompt


def get_answer(query: str) -> str:
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
    st.title("📑 RAG Document Assistant")
    st.caption("Ask questions about your document!")

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