import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
class Settings:
    BASE_URL=os.getenv("BASE_URL")
    CHROMADB_DIR = os.getenv("CHROMADB_DIR", "chromaDB")
    CHROMADB_COLLECTION = os.getenv("CHROMADB_COLLECTION", "demo001")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

settings = Settings()