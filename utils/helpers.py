import re
import logging
import sys


def normalize_text( text: str) -> str:
    
    text = re.sub(r'\s+', ' ', text)   # collapse whitespace
    text = re.sub(r'\n+', ' ', text)   # remove line breaks
    return text.strip()



# logging_config.py
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
