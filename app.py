import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load once using session state
@st.cache_resource
def load_rag():
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
            "k": 4,
            "fetch_k": 10,
            "lambda_multi": 0.5
        }
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant which answers questions.
        Answer based on the context given.
        If the question is out of context reply with
        'I couldn't find the answer in the document.'"""),
        ("human", "Context: {context}\nQuestion: {question}")
    ])
    return retriever, llm, prompt

# UI
st.title("📑RAG Document Assistant")
st.caption("Ask questions about your document!")

# Chat history
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
            retriever, llm, prompt = load_rag()

            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = llm.invoke(final_prompt)

        st.write(response.content)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.content
        })