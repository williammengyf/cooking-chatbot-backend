import re
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from .config import settings

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
# Initialize the vector store and retriever
# Updated to use the recommended import from langchain_ollama
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever()


store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


model = OllamaLLM(model=settings.llm_model)


def _remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert chef and a friendly conversational partner. "
            "Your role is to suggest a single, creative meal idea. "
            "Answer the user's question based ONLY on the following context. If the context doesn't contain the answer, say you don't have a relevant recipe. "
            "Do not add any 'chain of thought' or reasoning tags like `<think>`."
            "\n\n--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)


def log_and_pass(data, label=""):
    print(f"\n--- DEBUG LOG: {label} ---\n{data}\n--- END LOG ---\n")
    return data


base_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("message") | retriever,
    )
    | RunnableLambda(lambda x: log_and_pass(x, "1. After Retriever"))
    | rag_prompt
    | RunnableLambda(lambda x: log_and_pass(x, "2. After Prompt Formatting"))
    | model
    | RunnableLambda(lambda x: log_and_pass(x, "3. Raw Model Output"))
    | StrOutputParser()
    | RunnableLambda(_remove_think_tags)
)

chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="message",
    history_messages_key="history",
)
