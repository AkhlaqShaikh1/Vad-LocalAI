import uvicorn 
import logging
from server import app, HOST, PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleLocalAIService")

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT)
    logger.info(f"Server started at http://{HOST}:{PORT}")