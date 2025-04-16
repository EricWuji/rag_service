from fastapi import FastAPI
from app.lifespan import lifespan
from app.routes import router
from config import settings
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Async RAG Service",
    lifespan=lifespan
)
app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        loop="uvloop",
        http="httptools"
    )