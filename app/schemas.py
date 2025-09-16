from typing import List
from pydantic import BaseModel, Field


class MealSuggestionResponse(BaseModel):
    meal_name: str = Field(description="A name for the meal.")
    description: str = Field(description="A description of the meal.")
    ingredients_used: List[str] = Field(description="A list of the ingredients that are used in this meal.")


class ChatRequest(BaseModel):
    message: str = Field(description="The request from the user.")
