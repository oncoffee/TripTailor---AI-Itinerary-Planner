from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    lm_api_url: str
    lm_model_name: str

    class Config:
        env_file = ".env"


settings = Settings()
