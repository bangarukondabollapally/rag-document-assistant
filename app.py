import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Load RAG once (like cache)
# -----------------------------
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
        ("system",
         "You are an AI assistant which answers questions. "
         "Answer based on the context given. "
         "If the question is out of context reply with "
         "'I couldn't find the answer in the document.'"),
        ("human", "Context: {context}\nQuestion: {question}")
    ])

    return retriever, llm, prompt


retriever, llm, prompt = load_rag()


# -----------------------------
# Chat function (core logic)
# -----------------------------
def chat(message, history):
    docs = retriever.invoke(message)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": message
    })

    response = llm.invoke(final_prompt)

    return response.content


# -----------------------------
# Gradio UI
# -----------------------------
demo = gr.ChatInterface(
    fn=chat,
    title="📑 RAG Document Assistant",
    description="Ask questions about your document!"
)


demo.launch()