from fastapi import FastAPI
import logging
import src.configuration.config
from src.services.localAIservice import router as websocket_router

HOST = src.configuration.config.HOST
PORT = int(src.configuration.config.PORT)


logging.basicConfig(level=src.configuration.config.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimpleLocalAIService")

app = FastAPI(
    title="Simple Local Voice AI Service",
    description="A basic HTTP server to confirm Python environment setup.",
    version="0.0.1"
)


app.include_router(websocket_router)


@app.get("/")
async def read_root():
    logger.info("GET / endpoint accessed. Service is running.")
    return {"message": "Hello from Simple Local Voice AI Service!"}

