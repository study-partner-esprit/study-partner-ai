"""
Document tokenizer for chunking course documents into smaller pieces.

This module provides utilities to split large documents into manageable chunks
suitable for embedding and retrieval.
"""

from typing import List


class DocumentTokenizer:
    """
    Tokenizer that splits documents into overlapping chunks.

    This is useful for RAG (Retrieval Augmented Generation) where we need to:
    1. Split large documents into smaller chunks
    2. Create overlap between chunks to maintain context
    3. Keep chunks small enough for embedding models

    Args:
        chunk_size: Maximum number of characters per chunk (default: 500)
        overlap: Number of characters to overlap between chunks (default: 50)
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the tokenizer with chunk size and overlap.

        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of overlapping characters between consecutive chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def tokenize(self, documents: List[str]) -> List[str]:
        """
        Split documents into overlapping chunks.

        Process:
        1. For each document in the list
        2. Split it into chunks of size chunk_size
        3. Create overlap between consecutive chunks
        4. Return all chunks as a flat list

        Args:
            documents: List of document strings to tokenize

        Returns:
            List of text chunks suitable for embedding

        Example:
            >>> tokenizer = DocumentTokenizer(chunk_size=100, overlap=20)
            >>> docs = ["This is a long document that needs to be split..."]
            >>> chunks = tokenizer.tokenize(docs)
            >>> len(chunks)  # Number of chunks created
            5
        """
        all_chunks = []

        for document in documents:
            if not document or not document.strip():
                continue  # Skip empty documents

            # Split document into chunks with overlap
            chunks = self._chunk_text(document)
            all_chunks.extend(chunks)

        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split a single text into overlapping chunks.

        Uses a sliding window approach:
        - Window size = chunk_size
        - Step size = chunk_size - overlap

        Args:
            text: Single document string

        Returns:
            List of text chunks from this document
        """
        chunks = []
        text_length = len(text)

        # If text is shorter than chunk_size, return it as is
        if text_length <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # Calculate step size (how much to advance for each chunk)
        step = self.chunk_size - self.overlap

        # Create chunks using sliding window
        start = 0
        while start < text_length:
            # Get chunk from start to start + chunk_size
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end].strip()

            # Only add non-empty chunks
            if chunk:
                chunks.append(chunk)

            # Move to next position
            start += step

            # If we've reached the end, break
            if end >= text_length:
                break

        return chunks

    def get_chunk_info(self) -> dict:
        """
        Get information about the tokenizer configuration.

        Returns:
            Dictionary with tokenizer settings
        """
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "step_size": self.chunk_size - self.overlap,
        }
