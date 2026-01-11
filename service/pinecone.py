import os
from pinecone import Pinecone, ServerlessSpec
import uuid
from openai import AzureOpenAI
from utils.helpers import normalize_text
import nltk
nltk.download('punkt_tab')
import logging
logger = logging.getLogger(__name__)

class PineconeClient:
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
            try:
                to_upsert = []

                for chunk in chunks:
                    id = str(uuid.uuid4())
                    logger.info(f"Upserting chunk ID: {id}")
                    vector = self.get_embedding(chunk["chunk_text"])

                    to_upsert.append({
                        "id": id,
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
                logger.info(f"Upserted {len(to_upsert)} chunks to Pinecone index '{self.INDEX_NAME}' in namespace '{self.NAMESPACE}'.")
            except Exception as e:
                logger.error(f"Error upserting documents: {e}")
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

