from fastapi import FastAPI
from .schemas import ChatRequest, MealSuggestionResponse
from .services import chain

app = FastAPI(
    title="Cooking Chatbot API",
    description="An API that suggests meal ideas based on ingredients and preferences.",
    version="1.0.0"
)


@app.post("/chat", response_model=MealSuggestionResponse)
async def chat(request: ChatRequest):
    print(f"Received message: {request.message}")
    response = chain.invoke({"message": request.message})
    return response


@app.get("/")
async def root():
    return {"message": "Cooking Chatbot API is running"}
