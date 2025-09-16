from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, MealSuggestionResponse
from .services import chain

app = FastAPI(
    title="Cooking Chatbot API",
    description="An API that suggests meal ideas based on ingredients and preferences.",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",  # The origin of your Next.js app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.post("/chat", response_model=MealSuggestionResponse)
async def chat(request: ChatRequest):
    print(f"Received message: {request.message}")
    response = chain.invoke({"message": request.message})
    return response


@app.get("/")
async def root():
    return {"message": "Cooking Chatbot API is running"}
