import os
from pathlib import Path
from dotenv import load_dotenv
import logging
load_dotenv()

class Settings:
    BASE_URL=os.getenv("BASE_URL")
    CHROMADB_DIR = os.getenv("CHROMADB_DIR", "chromaDB")
    CHROMADB_COLLECTION = os.getenv("CHROMADB_COLLECTION", "demo001")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    API_KEY = os.getenv("API_KEY", "your_openai_api_key")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    INPUT_PDF = os.getenv("INPUT_PDF", "input/健康档案.pdf")
    TEXT_LANGUAGE = os.getenv("TEXT_LANGUAGE", "Chinese")
    CHUNK_SIZE = 500 if TEXT_LANGUAGE == "Chinese" else 1000
    CHUNK_OVERLAP = 100 if TEXT_LANGUAGE == "Chinese" else 200
    logger = logging.getLogger(__name__)

settings = Settings()