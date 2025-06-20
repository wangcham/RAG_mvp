# RAG_Manager class (main class, adjust imports as needed)
import logging
import os
import asyncio
from services.parser import FileParser
from services.chunker import Chunker
from services.embedder import Embedder
from services.retriever import Retriever
from services.database import create_db_and_tables, SessionLocal, KnowledgeBase, File, Chunk # Ensure Chunk is imported if you need to manipulate it directly
from services.embedding_models import EmbeddingModelFactory
from services.chroma_manager import ChromaIndexManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# RAG系统的整体流程操作类
class RAG_Manager:
    def __init__(self, embedding_model_type: str = "third_party_api",
                 embedding_model_name: str = "bge-m3"):
        self.logger = logging.getLogger(__name__)
        self.embedding_model_type = embedding_model_type
        self.embedding_model_name = embedding_model_name

        try:
            model = EmbeddingModelFactory.create_model(
                model_type=self.embedding_model_type,
                model_name_key=self.embedding_model_name
            )
            self.logger.info(f"Configured embedding model '{self.embedding_model_name}' has dimension: {model.embedding_dimension}")
        except Exception as e:
            self.logger.critical(f"Failed to get dimension for configured embedding model '{self.embedding_model_name}': {e}")
            raise RuntimeError("Failed to initialize RAG_Manager due to embedding model issues.")


        self.chroma_manager = ChromaIndexManager(collection_name=f"rag_collection_{self.embedding_model_name.replace('-', '_')}")

        self.parser = FileParser()
        self.chunker = Chunker()
        # Pass chroma_manager to Embedder and Retriever
        self.embedder = Embedder(
            model_type=self.embedding_model_type,
            model_name_key=self.embedding_model_name,
            chroma_manager=self.chroma_manager # Inject dependency
        )
        self.retriever = Retriever(
            model_type=self.embedding_model_type,
            model_name_key=self.embedding_model_name,
            chroma_manager=self.chroma_manager # Inject dependency
        )

    async def initialize_system(self):
        self.logger.info("Initializing database and Chroma index...")
        await asyncio.to_thread(create_db_and_tables)
        # Chroma collection is implicitly created on first access within ChromaIndexManager
        self.logger.info("System initialization complete.")

    async def store_data(self, file_path: str, kb_name: str, file_type: str, kb_description: str = "Default knowledge base"):
        self.logger.info(f"Starting data storage process for file: {file_path}")
        try:
            def _get_kb_sync(name):
                session = SessionLocal()
                try:
                    return session.query(KnowledgeBase).filter_by(name=name).first()
                finally:
                    session.close()

            kb = await asyncio.to_thread(_get_kb_sync, kb_name)

            if not kb:
                self.logger.info(f"Knowledge Base '{kb_name}' not found. Creating a new one.")
                def _add_kb_sync():
                    session = SessionLocal()
                    try:
                        new_kb = KnowledgeBase(name=kb_name, description=kb_description)
                        session.add(new_kb)
                        session.commit()
                        session.refresh(new_kb)
                        return new_kb
                    finally:
                        session.close()
                kb = await asyncio.to_thread(_add_kb_sync)
                self.logger.info(f"Created Knowledge Base: {kb.name} (ID: {kb.id})")

            def _add_file_sync(kb_id, file_name, path, file_type):
                session = SessionLocal()
                try:
                    file = File(kb_id=kb_id, file_name=file_name, path=path, file_type=file_type)
                    session.add(file)
                    session.commit()
                    session.refresh(file)
                    return file
                finally:
                    session.close()

            file_obj = await asyncio.to_thread(_add_file_sync, kb.id, os.path.basename(file_path), file_path, file_type)
            self.logger.info(f"Added file entry: {file_obj.file_name} (ID: {file_obj.id})")

            text = await self.parser.parse(file_path)
            if not text:
                self.logger.warning(f"File {file_path} parsed to empty content. Skipping chunking and embedding.")
                # You might want to delete the file_obj from the DB here if it's empty.
                session = SessionLocal()
                try:
                    session.delete(file_obj)
                    session.commit()
                except Exception as del_e:
                    self.logger.error(f"Failed to delete empty file_obj {file_obj.id}: {del_e}")
                finally:
                    session.close()
                return

            chunks_texts = await self.chunker.chunk(text)
            self.logger.info(f"Chunked into {len(chunks_texts)} pieces.")

            # embed_and_store now handles both DB chunk saving and Chroma embedding
            await self.embedder.embed_and_store(file_id=file_obj.id, chunks=chunks_texts)

            self.logger.info(f"Data storage process completed for file: {file_path}")

        except Exception as e:
            self.logger.error(f"Error in store_data for file {file_path}: {str(e)}", exc_info=True)
            # Consider cleaning up partially stored data if an error occurs.
            return

    async def retrieve_data(self, query: str):
        self.logger.info(f"Starting data retrieval process for query: '{query}'")
        try:
            retrieved_chunks = await self.retriever.retrieve(query)
            self.logger.info(f"Successfully retrieved {len(retrieved_chunks)} chunks for query.")
            return retrieved_chunks
        except Exception as e:
            self.logger.error(f"Error in retrieve_data for query '{query}': {str(e)}", exc_info=True)
            return []

    async def delete_data_by_file_id(self, file_id: int):
        """
        Deletes data associated with a specific file_id from both the relational DB and Chroma.
        """
        self.logger.info(f"Starting data deletion process for file_id: {file_id}")
        session = SessionLocal()
        try:
            # 1. Delete from Chroma
            await asyncio.to_thread(self.chroma_manager.delete_by_file_id_sync, file_id)

            # 2. Delete chunks from relational DB
            chunks_to_delete = session.query(Chunk).filter_by(file_id=file_id).all()
            for chunk in chunks_to_delete:
                session.delete(chunk)
            self.logger.info(f"Deleted {len(chunks_to_delete)} chunks from relational DB for file_id: {file_id}.")

            # 3. Delete file entry from relational DB
            file_to_delete = session.query(File).filter_by(id=file_id).first()
            if file_to_delete:
                session.delete(file_to_delete)
                self.logger.info(f"Deleted file entry {file_id} from relational DB.")
            else:
                self.logger.warning(f"File entry {file_id} not found in relational DB.")

            session.commit()
            self.logger.info(f"Data deletion completed for file_id: {file_id}.")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error deleting data for file_id {file_id}: {str(e)}", exc_info=True)
        finally:
            session.close()


# 测试用入口
async def main_run_standalone():
    rag_manager = RAG_Manager(
        embedding_model_type="third_party_api",
        embedding_model_name="bge-m3"
    )
    await rag_manager.initialize_system()

    test_file = "test_document_minilm.txt"
    file_content = "This is a document for MiniLM. It talks about efficient NLP models, such as bge-m3. It is a good starting point for learning about embeddings. This document provides more context on the topic of language models and their applications in various fields."
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(file_content)

    print("\n--- Storing Data ---")
    await rag_manager.store_data(
        file_path=test_file,
        kb_name="KB_NLP_Models",
        file_type="txt",
        kb_description="Knowledge Base about NLP models"
    )

    query = "What are efficient NLP models and their applications?"
    print(f"\n--- Retrieving Data for Query: '{query}' ---")
    results = await rag_manager.retrieve_data(query)
    print(f"\n--- Retrieval Results for Query: '{query}' ---")
    if results:
        for r in results:
            print(f"  Chunk ID: {r['chunk_id']}, File ID: {r.get('file_id')}, Distance: {r['distance']:.4f}, Text: {r['text'][:100]}...")
    else:
        print("No results found.")

    # Optional: Test deletion
    # Get the file_id of the test file to delete it
    session = SessionLocal()
    file_obj_to_delete = session.query(File).filter_by(file_name=os.path.basename(test_file)).first()
    session.close()

    if file_obj_to_delete:
        print(f"\n--- Deleting Data for File ID: {file_obj_to_delete.id} ---")
        await rag_manager.delete_data_by_file_id(file_obj_to_delete.id)
        # Verify deletion by trying to retrieve again (should return empty)
        print(f"\n--- Re-retrieving after deletion for Query: '{query}' ---")
        results_after_delete = await rag_manager.retrieve_data(query)
        if not results_after_delete:
            print("Successfully deleted data. No results found after deletion.")
        else:
            print("Deletion might have failed. Results still found.")

    if os.path.exists(test_file):
        os.remove(test_file)

    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main_run_standalone())