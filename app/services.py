import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama.llms import OllamaLLM

from .config import settings

store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


model = OllamaLLM(model=settings.llm_model)


def _remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert chef and a friendly conversational partner. "
            "Your role is to suggest a single, creative meal idea based on the user's request. "
            "Respond in a natural, helpful, and conversational tone. "
            "Your response must be direct and not include any 'chain of thought' or reasoning tags like `<think>`."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)

base_chain = (
    prompt
    | model
    | StrOutputParser()
    | RunnableLambda(_remove_think_tags)
)

chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="message",
    history_messages_key="history",
)
