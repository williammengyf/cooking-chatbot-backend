from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(description="The request from the user.")
    session_id: str = Field(description="A unique identifier for the user's session.")
