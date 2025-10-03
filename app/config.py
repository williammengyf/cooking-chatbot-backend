from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    base_llm_model: str = "qwen2-7b-instruct"
    finetuned_llm_model: str = "recipebot"


settings = Settings()
