# services/embedder.py
import asyncio
import logging
import numpy as np
from typing import List
from sqlalchemy.orm import Session # Import Session type hint
from services.base_service import BaseService # Assuming BaseService provides _run_sync
from services.database import Chunk, Vector, SessionLocal # Import SessionLocal
from services.faiss_manager import FAISSIndexManager
from services.embedding_models import  BaseEmbeddingModel, ThirdPartyAPIEmbeddingModel,EmbeddingModelFactory

logger = logging.getLogger(__name__)

class Embedder(BaseService): # Assuming BaseService defines _run_sync
    def __init__(self, model_type: str, model_name_key: str, faiss_manager: FAISSIndexManager):
        super().__init__() # Initialize BaseService, which should have _run_sync
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_type = model_type
        self.model_name_key = model_name_key # Use model_name_key consistently
        self.faiss_manager = faiss_manager # FAISS manager is injected

        # Load the specific embedding model
        # This is a synchronous call as part of __init__
        self.embedding_model: BaseEmbeddingModel = self._load_embedding_model()

        # Check for dimension mismatch immediately after model loading
        if self.embedding_model.embedding_dimension != self.faiss_manager.embedding_dim:
            self.logger.critical(
                f"FATAL: Embedder model dimension ({self.embedding_model.embedding_dimension}) "
                f"does not match FAISS manager dimension ({self.faiss_manager.embedding_dim}). "
                "This indicates a critical configuration error. Please ensure the FAISS index "
                "is built for the same model dimension used for embedding, or clear/rebuild the FAISS index."
            )
            raise ValueError("Embedding model and FAISS index dimension mismatch.")

    def _load_embedding_model(self) -> BaseEmbeddingModel:
        """
        Uses the factory to load the specific embedding model.
        This method remains synchronous as it's part of the Embedder's __init__.
        """
        self.logger.info(f"Loading embedding model: type={self.model_type}, name_key={self.model_name_key}...")
        try:
            # EmbeddingModelFactory.create_model is synchronous
            model = EmbeddingModelFactory.create_model(self.model_type, self.model_name_key)
            self.logger.info(f"Embedding model '{self.model_name_key}' loaded. Output dimension: {model.embedding_dimension}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load embedding model '{self.model_name_key}': {e}")
            raise

    def _db_save_chunks_and_embeddings_sync(self, session: Session, file_id: int, chunks_texts: List[str], embeddings_array: np.ndarray):
        """
        Synchronously saves chunks and their embeddings to the database and FAISS index.
        This function should be run in a separate thread via asyncio.to_thread.
        """
        self.logger.debug(f"Saving {len(chunks_texts)} chunks and embeddings for file_id {file_id} to DB and FAISS (sync).")
        
        chunk_objects = []
        for i, text in enumerate(chunks_texts):
            chunk = Chunk(file_id=file_id, text=text)
            session.add(chunk)
            chunk_objects.append(chunk)

        session.flush() # Use flush to get IDs without committing yet
        
        # Prepare data for FAISS
        chunk_ids = [chunk.id for chunk in chunk_objects]
        
        # Save vectors to the database
        self.faiss_manager.save_multiple_to_db_sync(session, chunk_ids, embeddings_array)
        
        session.commit() # Commit all changes at once (chunks and vectors)

        # Add vectors to the FAISS index (this is already designed to be synchronous in FAISSManager)
        self.faiss_manager.add_embeddings_sync(chunk_ids, embeddings_array)
        
        self.logger.debug(f"Successfully saved {len(chunk_objects)} chunks and embeddings to DB and FAISS (sync).")
        return chunk_objects

    async def embed_and_store(self, file_id: int, chunks: List[str]):
        """
        Asynchronously embeds text chunks and stores them in the database and FAISS.
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded. Please check Embedder initialization.")

        self.logger.info(f"Embedding {len(chunks)} chunks for file_id: {file_id} using {self.model_name_key}...")
        
        # Call the async embedding method of the model directly
        # This will use aiohttp for third-party APIs or call SentenceTransformer synchronously.
        embeddings: List[List[float]] = await self.embedding_model.embed_documents(chunks)
        
        # Convert list of lists to numpy array for FAISS compatibility
        embeddings_np = np.array(embeddings, dtype=np.float32)

        if embeddings_np.shape[1] != self.embedding_model.embedding_dimension:
            self.logger.error(f"Mismatch in embedding dimension: Model returned {embeddings_np.shape[1]}, expected {self.embedding_model.embedding_dimension}. Aborting storage.")
            raise ValueError("Embedding dimension mismatch during embedding process.")

        self.logger.info("Saving chunks and embeddings to database and FAISS...")
        
        # Pass SessionLocal() to the synchronous function which will manage its own session context
        saved_chunks = await self._run_sync(
            lambda fid, chks, embs_np: self._db_save_chunks_and_embeddings_sync(SessionLocal(), fid, chks, embs_np),
            file_id, chunks, embeddings_np
        )
        
        self.logger.info(f"Successfully saved {len(saved_chunks)} chunks and embeddings.")
        return saved_chunks