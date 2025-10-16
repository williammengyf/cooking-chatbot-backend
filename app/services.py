import re
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from .config import settings

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever()


store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


model = OllamaLLM(model=settings.finetuned_llm_model)


def _remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


general_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 “煮义煮义AI菜谱助手”，一位专业的厨师和友好的聊天伙伴。\n\n"
            "## 你的核心任务:\n"
            "1.  **直接回应**: 你的首要目标是根据用户的当前请求，提供一个完整、清晰、且实用的菜谱（包含食材和步骤）。\n"
            "2.  **保持专注**: 请专注于用户提出的具体问题。除非用户主动询问，否则不要推荐或提及任何其他无关的菜品。\n\n"
            "请用自然、友好的语气进行交流。"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 “煮义煮义AI菜谱助手”，一位专业的厨师。\n\n"
            "## 你的核心任务:\n"
            "你已经从内部知识库中检索到了以下上下文信息。请优先使用这些信息来回答用户的问题。\n\n"
            "--- 烹饪知识库 ---\n"
            "{context}\n"
            "--- 知识库结束 ---\n\n"
            "## 响应规则:\n"
            "1.  **评估相关性**: 首先判断知识库中的信息是否与用户的请求相关。\n"
            "2.  **优先使用知识库**: 如果信息相关，请严格依据知识库内容，为用户提供一个包含食材和步骤的完整食谱。\n"
            "3.  **优雅降级**: 如果知识库中的信息不相关，请礼貌地忽略它，并利用你自己的通用烹饪知识来直接回答用户的请求。\n"
            "4.  **保持专注**: 任何情况下，都不要推荐与用户当前请求无关的菜品。"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)


rag_chain = (
    rag_prompt
    | model
    | StrOutputParser()
    | RunnableLambda(_remove_think_tags)
)

general_chain = (
    general_prompt
    | model
    | StrOutputParser()
    | RunnableLambda(_remove_think_tags)
)

branch = RunnableBranch(
    (lambda x: x["context"], rag_chain),
    general_chain,
)

base_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("message") | retriever
    )
    | branch
)

chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="message",
    history_messages_key="history",
)
