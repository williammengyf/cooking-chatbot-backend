from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest
from .services import chain

app = FastAPI(
    title="Cooking Chatbot API",
    description="An API that suggests meal ideas based on ingredients and preferences.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        config = {"configurable": {"session_id": request.session_id}}
        response_string = await chain.ainvoke({"message": request.message}, config=config)
        return {"response": response_string}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Cooking Chatbot API is running"}
