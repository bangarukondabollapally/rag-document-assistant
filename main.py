from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":6,
        "fetch_k":20,
        "lambda_multi" : 0.7 # diversed results (0-high diversity, 1-low diversity)
    }
)

llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system","You are an ai assisstant which answers the question. Answer the question based on the context given. If any question is out of the context reply with couldn't find the answer"),
    ("human", "Context: {context}\nQuestion: {question}")
])

print("RAG System Created.\n\nPress 0 to exit\n")

while True:
    query = input("You: ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke(
        {
            "context":context,
            "question":query
        }
    )

    response = llm.invoke(final_prompt)

    print(f"\n RAG Response: {response.content}")