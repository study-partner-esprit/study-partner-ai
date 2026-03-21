"""Content retriever for RAG system."""

from typing import List, Optional
import numpy as np
from agents.planner.rag.tokenizer import DocumentTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


class ContentRetriever:
    """
    Content retriever that finds relevant concepts using vector similarity.

    Uses the vector store to find the most similar concepts to a query.
    Can also index course documents by tokenizing and embedding them.
    """

    def __init__(
        self, vector_store, embed_model, chunk_size: int = 800, overlap: int = 50
    ):
        """
        Initialize the content retriever.

        Args:
            vector_store: VectorStore instance for storing embeddings
            embed_model: EmbeddingModel for encoding queries
            chunk_size: Size of document chunks for tokenization (default: 500)
            overlap: Overlap between chunks (default: 50)
        """
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.tokenizer = DocumentTokenizer(chunk_size=chunk_size, overlap=overlap)
        self.indexed_chunks = []  # Keep track of indexed document chunks
        self.last_embeddings: Optional[np.ndarray] = None

    def add_precomputed_embeddings(
        self,
        chunks: List[str],
        embeddings: List[List[float]] | np.ndarray,
    ) -> int:
        """
        Add chunks with **pre-computed** embeddings directly to the vector store.

        This avoids re-encoding chunks that were already embedded at ingest time,
        saving model inference costs during planning.

        Args:
            chunks:     Text chunks matching the embeddings.
            embeddings: Pre-computed L2-normalised vectors — either
                        List[List[float]] or np.ndarray of shape (N, 384).

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0
        vec_matrix = np.array(embeddings, dtype="float32")
        self.vector_store.add(vec_matrix, chunks)
        self.indexed_chunks.extend(chunks)
        self.last_embeddings = vec_matrix
        logger.info(
            "retriever_precomputed_added",
            extra={"num_chunks": len(chunks)},
        )
        return len(chunks)

    def add_documents(
        self, documents: List[str], tokenization_settings: Optional[dict] = None
    ) -> int:
        """
        Add course documents to the knowledge base.

        Process:
        1. Tokenize documents into chunks using DocumentTokenizer
        2. Create embeddings for each chunk
        3. Add embeddings to the vector store
        4. Track chunks for retrieval

        Args:
            documents: List of document strings (can be OCR output)
            tokenization_settings: Optional dict with 'chunk_size' and 'overlap'
                                  to override defaults

        Returns:
            Number of chunks added to the knowledge base

        Example:
            >>> retriever.add_documents(["Course material about ML..."])
            15  # Returns number of chunks created and indexed
        """
        # Update tokenizer settings if provided
        if tokenization_settings:
            chunk_size = tokenization_settings.get(
                "chunk_size", self.tokenizer.chunk_size
            )
            overlap = tokenization_settings.get("overlap", self.tokenizer.overlap)
            self.tokenizer = DocumentTokenizer(chunk_size=chunk_size, overlap=overlap)

        # Tokenize documents into chunks
        chunks = self.tokenizer.tokenize(documents)

        if not chunks:
            return 0

        # Create embeddings for all chunks
        try:
            chunk_embeddings = self.embed_model.encode(chunks)

            # Add embeddings and chunks to vector store
            self.vector_store.add(chunk_embeddings, chunks)

            # Track the chunks for later reference
            self.indexed_chunks.extend(chunks)
            self.last_embeddings = chunk_embeddings

            return len(chunks)

        except Exception as e:
            logger.warning("retriever_add_documents_error", extra={"error": str(e)})
            return 0

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve the most relevant concepts for a given query.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of relevant concept strings
        """
        # Return empty list if no documents are indexed
        if not self.indexed_chunks:
            return []

        try:
            # Encode the query
            query_embedding = self.embed_model.encode([query])[0]

            # Search the vector store
            results = self.vector_store.search(query_embedding.reshape(1, -1), k=top_k)

            return results if results else []

        except Exception as e:
            logger.warning("retriever_retrieve_error", extra={"error": str(e)})
            return []

        """
        Add course documents to the knowledge base.

        Process:
        1. Tokenize documents into chunks using DocumentTokenizer
        2. Create embeddings for each chunk
        3. Add embeddings to the vector store
        4. Track chunks for retrieval

        Args:
            documents: List of document strings (can be OCR output)
            tokenization_settings: Optional dict with 'chunk_size' and 'overlap'
                                  to override defaults

        Returns:
            Number of chunks added to the knowledge base

        Example:
            >>> retriever.add_documents(["Course material about ML..."])
            15  # Returns number of chunks created and indexed
        """
        # Update tokenizer settings if provided
        if tokenization_settings:
            chunk_size = tokenization_settings.get(
                "chunk_size", self.tokenizer.chunk_size
            )
            overlap = tokenization_settings.get("overlap", self.tokenizer.overlap)
            self.tokenizer = DocumentTokenizer(chunk_size=chunk_size, overlap=overlap)

        # Tokenize documents into chunks
        chunks = self.tokenizer.tokenize(documents)

        if not chunks:
            return 0

        # Create embeddings for all chunks
        try:
            chunk_embeddings = self.embed_model.encode(chunks)

            # Add embeddings and chunks to vector store
            self.vector_store.add(chunk_embeddings, chunks)

            # Track the chunks for later reference
            self.indexed_chunks.extend(chunks)

            return len(chunks)

        except Exception as e:
            print(f"Error indexing documents: {e}")
            return 0

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve the most relevant concepts for a given query.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of relevant concept strings
        """
        # Return empty list if no documents are indexed
        if not self.indexed_chunks:
            return []

        try:
            # Encode the query
            query_embedding = self.embed_model.encode([query])[0]

            # Search the vector store
            results = self.vector_store.search(query_embedding.reshape(1, -1), k=top_k)

            return results if results else []

        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
