from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def setup_rag():
    """Initialize and return the retriever, LLM, and prompt template."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_multi": 0.7  # diverse results (0-high diversity, 1-low diversity)
        }
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant which answers the question. Answer the question based on the context given. If any question is out of the context reply with couldn't find the answer"),
        ("human", "Context: {context}\nQuestion: {question}")
    ])

    return retriever, llm, prompt


def get_answer(retriever, llm, prompt, query: str) -> str:
    """Retrieve relevant context and generate an answer for the given query."""
    docs = retriever.invoke(query)
    context = "".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)
    return response.content


def run_chat_loop():
    """Run the interactive Q&A loop in the terminal."""
    load_dotenv()
    retriever, llm, prompt = setup_rag()

    print("RAG System Created.\n\nPress 0 to exit\n")

    while True:
        query = input("You: ")
        if query == "0":
            break

        answer = get_answer(retriever, llm, prompt, query)
        print(f"\nRAG Response: {answer}")


if __name__ == "__main__":
    run_chat_loop()