import asyncio
import logging
import numpy as np
from typing import List, Dict, Any
from sqlalchemy.orm import Session 
from services.base_service import BaseService 
from services.database import Chunk, SessionLocal 
from services.faiss_manager import FAISSIndexManager
from services.embedding_models import BaseEmbeddingModel, EmbeddingModelFactory

logger = logging.getLogger(__name__)

class Retriever(BaseService): 
    def __init__(self, model_type: str, model_name_key: str, faiss_manager: FAISSIndexManager):
        super().__init__() 
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_type = model_type
        self.model_name_key = model_name_key # Use model_name_key consistently
        self.faiss_manager = faiss_manager # FAISS manager is injected

        # Load the specific embedding model
        self.embedding_model: BaseEmbeddingModel = self._load_embedding_model()

        # Check for dimension mismatch immediately after model loading
        if self.embedding_model.embedding_dimension != self.faiss_manager.embedding_dim:
            self.logger.critical(
                f"FATAL: Retriever model dimension ({self.embedding_model.embedding_dimension}) "
                f"does not match FAISS manager dimension ({self.faiss_manager.embedding_dim}). "
                "This indicates a critical configuration error. Please ensure the FAISS index "
                "is built for the same model dimension used for retrieval."
            )
            raise ValueError("Retriever model and FAISS index dimension mismatch.")

    def _load_embedding_model(self) -> BaseEmbeddingModel:
        """
        Uses the factory to load the specific embedding model.
        This method remains synchronous as it's part of the Retriever's __init__.
        """
        self.logger.info(f"Loading retriever embedding model: type={self.model_type}, name_key={self.model_name_key}...")
        try:
            
            model = EmbeddingModelFactory.create_model(self.model_type, self.model_name_key)
            self.logger.info(f"Retriever embedding model '{self.model_name_key}' loaded. Output dimension: {model.embedding_dimension}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load retriever embedding model '{self.model_name_key}': {e}")
            raise

    # Synchronous method to handle FAISS search (from FAISSIndexManager)
    def _faiss_search_sync(self, query_embedding: np.ndarray, k: int):
        """
        Performs a synchronous FAISS search.
        This function should be run in a separate thread via asyncio.to_thread.
        """
        # query_embedding needs to be reshaped to (1, -1) for single query search
        return self.faiss_manager.search_sync(query_embedding.reshape(1, -1), k)

    # Synchronous method to fetch chunk content from the database
    def _db_get_chunks_sync(self, session: Session, chunk_ids: List[int]) -> List[Chunk]:
        """
        Synchronously fetches chunk objects from the database given their IDs.
        This function should be run in a separate thread via asyncio.to_thread.
        """
        self.logger.debug(f"Fetching {len(chunk_ids)} chunk details from database (sync).")
        chunks = session.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
        return chunks

    async def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieves the most relevant text chunks for a given query.
        1. Embed the query using the configured embedding model.
        2. Search in FAISS for relevant chunk IDs and distances.
        3. Fetch the full chunk content from the database.
        """
        if not self.embedding_model:
            raise RuntimeError("Retriever embedding model not loaded. Please check Retriever initialization.")

        self.logger.info(f"Retrieving for query: '{query}' with k={k} using {self.model_name_key}")

        # 1. Embed the query (asynchronously)
        # self.embedding_model.embed_query is now an async method
        query_embedding: List[float] = await self.embedding_model.embed_query(query)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)

        # 2. Search in FAISS (synchronously in a thread)
        distances, chunk_ids_np = await self._run_sync(self._faiss_search_sync, query_embedding_np, k)

        if chunk_ids_np.size == 0 or (chunk_ids_np == -1).all():
            self.logger.info("No valid chunks retrieved from FAISS.")
            return []

        # Filter out -1 (invalid) IDs and flatten the array
        valid_chunk_ids = [int(cid) for cid in chunk_ids_np.flatten() if cid != -1]
        
        if not valid_chunk_ids:
            self.logger.info("No valid chunk IDs found after FAISS search filter.")
            return []

        # 3. Fetch chunk content from the database (synchronously in a thread)
        self.logger.info(f"Fetching {len(valid_chunk_ids)} chunk details from database...")
        
        # Pass SessionLocal() to the synchronous function which will manage its own session context
        chunks_from_db = await self._run_sync(
            lambda cids: self._db_get_chunks_sync(SessionLocal(), cids),
            valid_chunk_ids
        )

        # Map chunk IDs to their objects for easy lookup
        chunk_map = {chunk.id: chunk for chunk in chunks_from_db}
        
        results: List[Dict[str, Any]] = []
        # Iterate through the original FAISS results to maintain order and associate distances
        for i, chunk_id_result in enumerate(chunk_ids_np.flatten()):
            if chunk_id_result != -1 and int(chunk_id_result) in chunk_map:
                chunk_obj = chunk_map[int(chunk_id_result)]
                results.append({
                    "chunk_id": chunk_obj.id,
                    "text": chunk_obj.text,
                    "distance": float(distances.flatten()[i]) # Ensure distance is float
                })
            elif chunk_id_result != -1:
                self.logger.warning(f"Chunk ID {chunk_id_result} found in FAISS but not in DB. Data inconsistency detected.")
        
        self.logger.info(f"Retrieved {len(results)} chunks.")
        return results