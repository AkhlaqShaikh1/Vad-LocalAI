import os
from dotenv import load_dotenv

#load environment variables from .env file
load_dotenv()

# Get the host and port from environment variables
HOST = "0.0.0.0"
PORT = 8080

# logging configuration
LOG_LEVEL = "INFO"