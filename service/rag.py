import os
from pinecone import Pinecone, ServerlessSpec
import uuid
from openai import AzureOpenAI
from utils.helpers import normalize_text
import nltk
nltk.download('punkt_tab')
import logging
logger = logging.getLogger(__name__)
class RAGClient:
    INDEX_NAME = "dense-index"                       
    NAMESPACE = "acmecloud_documents"    


    def __init__(self):
        # Initialize Pinecone client and index
        try:
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            # Azure OpenAI client for embeddings
            self.azure_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview"          # or newer version
            )
            self.embedding_deployment = "text-embedding-3-small"  

            self.index = self.init_pinecone_index()

            logger.info("RAGClient initialized with Pinecone index '%s' and namespace '%s'.", self.INDEX_NAME, self.NAMESPACE)
        except Exception as e:
            logger.error(f"Error initializing RAGClient: {e}")
            raise e
    def init_pinecone_index(self):
        """
        Initialize Pinecone and create an index using the integrated embedding model.
        The integrated model will automatically generate embeddings from the field 'chunk_text'.
        """

        if self.INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.INDEX_NAME,
                dimension=1536,                     # ← very important!
                metric="cosine",                    # or "euclidean"
                spec=ServerlessSpec(
                    cloud="aws",                    # or "azure", "gcp"
                    region="us-east-1"              # choose closest/fastest
                )
            )

        index = self.pc.Index(self.INDEX_NAME)
        return index

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding using your Azure OpenAI model"""
        try:
            response = self.azure_client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise e

    def upsert_documents(self, chunks: list[dict]):
            """
            Example chunk format:
            {
                "id": "unique-id",
                "chunk_text": "...",
                "metadata": {"source": "doc.pdf", "page": 3}
            }
            """
            to_upsert = []

            for chunk in chunks:
                vector = self.get_embedding(chunk["chunk_text"])

                to_upsert.append({
                    "id": chunk.get("id", str(uuid.uuid4())),
                    "values": vector,
                    "metadata": {
                        "text": chunk["chunk_text"],        
                        **chunk.get("metadata", {})
                    }
                })

            self.index.upsert(
                vectors=to_upsert,
                namespace=self.NAMESPACE
            )



    def generate_chunks(self, max_chars: int = 1000) -> list[dict]:
        """
        Reads all PDF files in the '../documents' folder, extracts and cleans their text,
        and splits the text into smaller chunks of a specified maximum character length.

        Each chunk contains a portion of text that does not exceed `max_chars` and 
        preserves sentence boundaries. Metadata is attached to each chunk to indicate
        its source PDF file.

        Parameters:
        -----------
        max_chars : int, optional (default=1000)
            The maximum number of characters allowed in a single chunk. 
            Sentences are grouped until this limit is reached, then a new chunk is created.

        Returns:
        --------
        list of dict
            A list of dictionaries, each representing a text chunk. Each dictionary has:
            - 'chunk_text': str
                The extracted text content of the chunk.
            - 'metadata': dict
                A dictionary containing metadata about the chunk, currently:
                - 'source': str
                    The filename of the PDF from which the chunk was extracted.
        """
        from pypdf import PdfReader
        from nltk.tokenize import sent_tokenize

        try:
            documents_folder = "documents"
            pdf_files = [f for f in os.listdir(f"{documents_folder}") if f.endswith(".pdf")]

            chunks = []

            for pdf_file in pdf_files:
                pdf_path = os.path.join(f"{documents_folder}", pdf_file)
                reader = PdfReader(pdf_path)

                raw_text = ""
                for page in reader.pages:
                    raw_text += page.extract_text() + " "

                clean_text = normalize_text(raw_text)
                sentences = sent_tokenize(clean_text)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chars:
                        current_chunk += " " + sentence
                    else:
                        chunks.append({
                            "chunk_text": current_chunk.strip(),
                            "metadata": {
                                "source": pdf_file
                            }
                        })
                        current_chunk = sentence

                if current_chunk:
                    chunks.append({
                        "chunk_text": current_chunk.strip(),
                        "metadata": {
                            "source": pdf_file
                        }
                    })

            return chunks
        except Exception as e:
            logger.error(f"Error generating chunks: {e}")
            raise e

    def search_similar_chunks(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for similar chunks in Pinecone index"""
        try:
            query_embedding = self.get_embedding(query)

            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.NAMESPACE
            )

            results = []
            for match in response.matches:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            return results
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            raise e

