import uvicorn 
import logging
from src.server import app, HOST, PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleLocalAIService")

if __name__ == "__main__":
    uvicorn.run("src.server:app", host=HOST, port=PORT, reload=True)
    logger.info(f"Server started at http://{HOST}:{PORT}")