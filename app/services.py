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


base_model = OllamaLLM(model=settings.base_llm_model)
finetuned_model = OllamaLLM(model=settings.finetuned_llm_model)


def _remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


general_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位专业且富有创意的厨师。请仔细理解用户的整个对话历史，确保你的推荐符合用户当前和之前的所有要求。\n\n"
            "重要指导原则：\n"
            "1. **上下文记忆**：必须记住用户之前提到的偏好、限制和排除要求\n"
            "2. **创意多样性**：避免重复相似的食谱结构，根据用户需求提供真正不同的烹饪方法\n"
            "3. **个性化响应**：针对用户的具体要求进行定制，而不是提供通用模板\n"
            "4. **明确排除**：如果用户明确排除了某些食材或做法，绝对不要再次包含\n\n"
            "请确保你的回应：\n"
            "- 首先确认理解用户的需求变化\n"
            "- 提供有创意的、不同于之前建议的菜品\n"
            "- 包含清晰的食材清单和步骤\n"
            "- 用友好、对话式的语气交流"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位专业的厨师。请严格基于以下烹饪知识库，并仔细考虑整个对话历史来回答用户的问题。\n\n"
            "--- 烹饪知识库 ---\n{context}\n--- 知识库结束 ---\n\n"
            "关键响应原则：\n"
            "1. **上下文一致性**：确保推荐与用户之前的所有要求一致\n"
            "2. **知识库优先**：主要从提供的知识库中寻找解决方案\n"
            "3. **需求跟踪**：明确跟踪用户需求的变化轨迹\n"
            "4. **创意约束**：在用户约束范围内提供创新方案\n\n"
            "响应结构：\n"
            "1. 先确认理解用户的最新需求和历史偏好\n"
            "2. 明确说明参考了知识库中的哪些内容\n"
            "3. 提供完全符合用户所有要求的定制化食谱\n"
            "4. 如果知识库限制无法满足需求，诚实地说明限制\n\n"
            "特别注意：绝对不要推荐用户已经明确排除的食材或烹饪方法。"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)


def log_and_pass(data, label=""):
    print(f"\n--- DEBUG LOG: {label} ---\n{data}\n--- END LOG ---\n")
    return data


# 1. Define the RAG path: uses the fine-tuned model
rag_chain = (
    rag_prompt
    | finetuned_model
    | StrOutputParser()
    | RunnableLambda(_remove_think_tags)
)

# 2. Define the General path: uses the powerful base model
general_chain = (
    general_prompt
    | base_model
    | StrOutputParser()
    | RunnableLambda(_remove_think_tags)
)

# 3. Build the branching logic
branch = RunnableBranch(
    (lambda x: x["context"], rag_chain),
    general_chain,
)

# 4. Construct the full chain
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
