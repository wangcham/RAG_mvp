# services/faiss_manager.py
import faiss
import numpy as np
import asyncio
import logging
from services.database import SessionLocal, Vector, Chunk

class FAISSIndexManager:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunk_ids = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"FAISSIndexManager initialized with dimension: {self.embedding_dim}")


    def add_embeddings_sync(self, chunk_ids: list[int], embeddings: np.ndarray):
        if embeddings.size > 0:
            if embeddings.shape[1] != self.embedding_dim:
                self.logger.error(f"Attempted to add embeddings of dimension {embeddings.shape[1]} to FAISS index with dimension {self.embedding_dim}.")
                raise ValueError("Embedding dimension mismatch when adding to FAISS index.")
            self.index.add(embeddings)
            self.chunk_ids.extend(chunk_ids)

    def search_sync(self, query_embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self.index.ntotal == 0:
            self.logger.warning("FAISS index is empty, returning no results.")
            return np.array([]), np.array([])

        if query_embedding.shape[1] != self.embedding_dim:
            self.logger.error(f"Query embedding dimension {query_embedding.shape[1]} does not match FAISS index dimension {self.embedding_dim}.")
            raise ValueError("Query embedding dimension mismatch for FAISS search.")

        distances, indices = self.index.search(query_embedding, k)
        valid_indices_mask = indices != -1
        valid_indices = indices[valid_indices_mask]
        valid_distances = distances[valid_indices_mask]

        retrieved_chunk_ids = [self.chunk_ids[idx] for idx in valid_indices.flatten()]
        return valid_distances, np.array(retrieved_chunk_ids)

    def load_from_db_sync(self):
        session = SessionLocal()
        try:
            vectors_from_db = session.query(Vector).all()
            if not vectors_from_db:
                self.logger.info("No vectors found in DB to load into FAISS index.")
                self.index = faiss.IndexFlatL2(self.embedding_dim) # Ensure index is initialized
                self.chunk_ids = []
                return

            embeddings_list = []
            chunk_ids_list = []
            for v in vectors_from_db:
                embeddings_list.append(np.frombuffer(v.embedding, dtype=np.float32))
                chunk_ids_list.append(v.chunk_id)

            if embeddings_list:
                embeddings_array = np.array(embeddings_list)
                if embeddings_array.shape[1] != self.embedding_dim:
                    self.logger.warning(
                        f"Loaded embedding dimension {embeddings_array.shape[1]} from DB does not match current FAISS index dimension {self.embedding_dim}. "
                        "This indicates a potential mismatch of models used for embedding. "
                        "Rebuilding index with the dimension from loaded data, but consider if this is intended."
                    )
                    # If dimensions don't match, we assume the loaded data's dimension is authoritative for this load
                    # This means the FAISS index itself must be re-created with the new dimension
                    self.embedding_dim = embeddings_array.shape[1]
                    self.index = faiss.IndexFlatL2(self.embedding_dim) # Re-initialize with new dimension
                else:
                    self.index = faiss.IndexFlatL2(self.embedding_dim) # Re-initialize with existing dimension

                self.index.add(embeddings_array)
                self.chunk_ids = chunk_ids_list
                self.logger.info(f"Loaded {len(embeddings_list)} vectors into FAISS index. Index dimension: {self.embedding_dim}")
            else:
                self.logger.info("No valid embeddings to load into FAISS index after filtering.")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.chunk_ids = []
        finally:
            session.close()

    def save_to_db_sync(self, session, chunk_id: int, embedding: np.ndarray):
        existing_vector = session.query(Vector).filter(Vector.chunk_id == chunk_id).first()
        if existing_vector:
            existing_vector.embedding = embedding.tobytes()
        else:
            new_vector = Vector(chunk_id=chunk_id, embedding=embedding.tobytes())
            session.add(new_vector)
        session.commit()

    def save_multiple_to_db_sync(self, session, chunk_ids: list[int], embeddings: list[np.ndarray]):
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            existing_vector = session.query(Vector).filter(Vector.chunk_id == chunk_id).first()
            if existing_vector:
                existing_vector.embedding = embedding.tobytes()
            else:
                new_vector = Vector(chunk_id=chunk_id, embedding=embedding.tobytes())
                session.add(new_vector)
        session.commit()
