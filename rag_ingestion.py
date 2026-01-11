from dotenv import load_dotenv
load_dotenv()
import nltk
nltk.download('punkt_tab')
import os
import logging
from pypdf import PdfReader
from service.pinecone import PineconeClient
import uuid
from utils.helpers import normalize_text, setup_logging
setup_logging()
logger = logging.getLogger(__name__)

DOCUMENTS_DIR =  "documents"
MAX_CHARS = 1000


def extract_text_from_pdfs(folder: str) -> list[dict]:
    """Load and clean all PDFs from a folder"""
    documents = []
    logger.info(f"Extracting text from PDFs in folder: {folder}")
    for filename in os.listdir(folder):
        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(folder, filename)
        logger.info(f"Reading PDF: {filename}")

        reader = PdfReader(path)
        raw_text = ""

        for page in reader.pages:
            raw_text += (page.extract_text() or "") + " "

        documents.append({
            "text": normalize_text(raw_text),
            "source": filename
        })

    logger.info(f"Loaded {len(documents)} documents.")
    return documents

def chunk_documents(documents: list[dict], max_chars: int) -> list[dict]:
    """Sentence-aware chunking"""
    from nltk.tokenize import sent_tokenize

    chunks = []
    logger.info("Starting document chunking.")
    for doc in documents:
        current_chunk = ""
        sentences = sent_tokenize(doc["text"])

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += " " + sentence
            else:
                chunks.append({
                    "chunk_text": current_chunk.strip(),
                    "metadata": {"source": doc["source"]}
                })
                current_chunk = sentence

        if current_chunk:
            chunks.append({
                "chunk_text": current_chunk.strip(),
                "metadata": {"source": doc["source"]}
            })

    logger.info(f"Created {len(chunks)} chunks.")
    return chunks



def main():
    logger.info("Starting document ingestion")

    pc = PineconeClient()

    documents = extract_text_from_pdfs(DOCUMENTS_DIR)
    chunks = chunk_documents(documents, MAX_CHARS)

    pc.upsert_documents(chunks)

    logger.info("Ingestion completed successfully")


if __name__ == "__main__":
    main()
