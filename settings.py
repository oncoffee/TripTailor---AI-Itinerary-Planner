from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    lm_api_url: str
    lm_model_name: str
    milvus_host: str
    milvus_port: str

    class Config:
        env_file = ".env"


settings = Settings()
