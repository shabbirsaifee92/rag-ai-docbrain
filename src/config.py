from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    EMBEDDINGS_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DB_PATH: str = "./chroma_db"
    K_DOCUMENTS: int = 3

    # LLM Settings
    LLM_BASE_URL: str = "http://localai:8080"
    LLM_MODEL_NAME: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.9
    MAX_TOKENS: int = 2048

    class Config:
        env_file = ".env"

settings = Settings()
