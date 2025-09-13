from fastapi import FastAPI
from pydantic import BaseModel

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
    return {"response": response}

@app.get("/")
async def root():
    return {"message": "Welcome to the Chat API!"}

