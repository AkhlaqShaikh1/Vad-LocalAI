import logging
import sys
from datetime import datetime
from colorama import init, Fore, Style


init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.sent_event_counter = 0
        self.received_event_counter = -1
    
    def format(self, record):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = record.getMessage()
        origin = message.split(' ', 1)[0] if message else ''
        
        if origin == "[Client]":
            counter = f"C-{self.sent_event_counter:04d}"
            self.sent_event_counter += 1
            colored_message = f"{Fore.CYAN}{message}{Style.RESET_ALL}"
        elif origin == "[Server]":
            counter = f"S-{self.received_event_counter:04d}"
            self.received_event_counter += 1
            colored_message = f"{Fore.YELLOW}{message}{Style.RESET_ALL}"
        else:
            counter = "N/A"
            colored_message = f"{Fore.WHITE}{Style.DIM}{message}{Style.RESET_ALL}"
        
        return f"{counter} | {timestamp} [{record.levelname}] {colored_message}"


logger = logging.getLogger('voice_ai_logger')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
formatter = ColoredFormatter()
console_handler.setFormatter(formatter)


logger.addHandler(console_handler)


def log_client(msg):
    """Log client-side events"""
    logger.info(f"[Client] {msg}")

def log_server(msg):
    """Log server-side events"""
    logger.info(f"[Server] {msg}")

def get_counters():
    """Get current counter values"""
    return {
        'sent_event_counter': formatter.sent_event_counter,
        'received_event_counter': formatter.received_event_counter
    }