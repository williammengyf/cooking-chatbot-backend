from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{message}")
])

model = OllamaLLM(model="deepseek-r1:1.5b")

chain = template | model

class ChatRequest(BaseModel):
    message: str

app = FastAPI(
    title="Chat API",
    description="A simple chat API using FastAPI",
    version="1.0.0"
)

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"Received message: {request.message}")
    response = chain.invoke({"message": request.message})
    return {"response": response}

@app.get("/")
async def root():
    return {"message": "Welcome to the Chat API!"}

