import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

if __name__ == "__main__":
    from src.main import app, HOST, PORT
    import uvicorn
    uvicorn.run("src.server:app", host=HOST, port=PORT, reload=True)

