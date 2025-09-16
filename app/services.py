from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM

from .config import settings
from .schemas import MealSuggestionResponse

model = OllamaLLM(model=settings.llm_model)

parser = PydanticOutputParser(pydantic_object=MealSuggestionResponse)

template = """
You are an expert chef.
Suggest a meal idea based on the following ingredients and preferences.

{format_instructions}

User's request: {message}
"""

prompt = ChatPromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
