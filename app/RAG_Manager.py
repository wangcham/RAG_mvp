import logging
import os
import asyncio
from services.parser import FileParser
from services.chunker import Chunker
from services.embedder import Embedder
from services.retriever import Retriever
from services.database import create_db_and_tables, SessionLocal, KnowledgeBase, File
from services.faiss_manager import FAISSIndexManager
from services.embedding_models import ThirdPartyAPIEmbeddingModel, EMBEDDING_MODEL_CONFIGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# RAG系统的整体流程操作类
class RAG_Manager:
    def __init__(self, embedding_model_type: str = "third_party_api",
                 embedding_model_name: str = "bge-m3"):
        self.logger = logging.getLogger(__name__)
        self.embedding_model_type = embedding_model_type
        self.embedding_model_name = embedding_model_name

        # 定位嵌入模型
        try:
            dummy_model = ThirdPartyAPIEmbeddingModel(
                model_name=self.embedding_model_name,
                api_endpoint=EMBEDDING_MODEL_CONFIGS[self.embedding_model_name]['api_endpoint'],
                headers=EMBEDDING_MODEL_CONFIGS[self.embedding_model_name]['headers'],
                payload_template=EMBEDDING_MODEL_CONFIGS[self.embedding_model_name]['payload_template'],
                embedding_dimension=EMBEDDING_MODEL_CONFIGS[self.embedding_model_name]['embedding_dimension']
            )
            self.expected_embedding_dim = dummy_model._embedding_dimension
            self.logger.info(f"Configured embedding model '{self.embedding_model_name}' has dimension: {self.expected_embedding_dim}")
        except Exception as e:
            self.logger.critical(f"Failed to get dimension for configured embedding model '{self.embedding_model_name}': {e}")
            raise RuntimeError("Failed to initialize RAG_Manager due to embedding model issues.")

        # faiss索引
        self.faiss_manager = FAISSIndexManager(self.expected_embedding_dim)

        self.parser = FileParser()
        self.chunker = Chunker()
        
        # 定位嵌入器和检索器，但是后续需要确保模型一致
        self.embedder = Embedder(model_type=self.embedding_model_type,
                                 model_name_key=self.embedding_model_name,
                                 faiss_manager=self.faiss_manager)
        self.retriever = Retriever(model_type=self.embedding_model_type,
                                  model_name_key=self.embedding_model_name,
                                  faiss_manager=self.faiss_manager)

    async def initialize_system(self):
        # 启动！
        """Initializes database tables and loads FAISS index."""
        self.logger.info("Initializing database and FAISS index...")
        await asyncio.to_thread(create_db_and_tables)

        
        await asyncio.to_thread(self.faiss_manager.load_from_db_sync)

        if self.faiss_manager.embedding_dim != self.expected_embedding_dim:
            self.logger.critical(
                f"FATAL: FAISS index dimension ({self.faiss_manager.embedding_dim}) "
                f"does not match configured embedding model dimension ({self.expected_embedding_dim}). "
                "This means vectors in your database were created with a different model. "
                "You must either clear your database (vectors table) or switch to the correct model."
            )
            raise ValueError("FAISS index dimension mismatch with configured embedding model.")

        self.logger.info("System initialization complete.")

    async def store_data(self, file_path: str, kb_name: str, file_type: str, kb_description: str = "Default knowledge base"):
        """
        存储数据：解析文件 -> 分块 -> 嵌入并存储。
        """
        self.logger.info(f"Starting data storage process for file: {file_path}")
        try:
            # 内部同步函数：获取知识库
            def _get_kb_sync(name):
                session = SessionLocal() # 在新线程中创建会话
                try:
                    return session.query(KnowledgeBase).filter_by(name=name).first()
                finally:
                    session.close() # 确保会话关闭
            
            # 在单独的线程中运行同步的数据库查询
            kb = await asyncio.to_thread(_get_kb_sync, kb_name)

            if not kb:
                self.logger.info(f"Knowledge Base '{kb_name}' not found. Creating a new one.")
                # 内部同步函数：添加知识库
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

            # 内部同步函数：添加文件
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

            # 在单独的线程中运行同步的数据库操作
            file_obj = await asyncio.to_thread(_add_file_sync, kb.id, os.path.basename(file_path), file_path, file_type)
            self.logger.info(f"Added file entry: {file_obj.file_name} (ID: {file_obj.id})")

            # 文件解析 
            text = await self.parser.parse(file_path)
            if not text:
                self.logger.warning(f"File {file_path} parsed to empty content. Skipping chunking and embedding.")
                return

            chunks = await self.chunker.chunk(text)
            self.logger.info(f"Chunked into {len(chunks)} pieces.")

            await self.embedder.embed_and_store(file_id=file_obj.id, chunks=chunks)

            self.logger.info(f"Data storage process completed for file: {file_path}")

        except Exception as e:
            self.logger.error(f"Error in store_data for file {file_path}: {str(e)}", exc_info=True)
            return

    async def retrieve_data(self, query: str):
        """
        检索过程：嵌入查询 -> 检索相关分块。
        """
        self.logger.info(f"Starting data retrieval process for query: '{query}'")
        try:
            retrieved_chunks = await self.retriever.retrieve(query)
            self.logger.info(f"Successfully retrieved {len(retrieved_chunks)} chunks for query.")
            return retrieved_chunks
        except Exception as e:
            self.logger.error(f"Error in retrieve_data for query '{query}': {str(e)}", exc_info=True)
            return []

# 测试你的
async def main_run_standalone():
    rag_manager_minilm = RAG_Manager(
        embedding_model_type="third_party_api",
        embedding_model_name="bge-m3"
    )
    await rag_manager_minilm.initialize_system()

    dummy_file_path_minilm = "test_document_minilm.txt"
    with open(dummy_file_path_minilm, "w", encoding="utf-8") as f:
        f.write("This is a document for MiniLM. It talks about efficient NLP models,such as bgm-m3. It is a good starting point for learning about embeddings.")
    await rag_manager_minilm.store_data(
        file_path=dummy_file_path_minilm,
        kb_name="KB",
        file_type="txt"
    )
    query_minilm = "What are efficient NLP models?"
    results_minilm = await rag_manager_minilm.retrieve_data(query_minilm)
    print(f"\n--- Retrieval Results for Query: '{query_minilm}' (MiniLM) ---")
    for r in results_minilm:
        print(f"  Chunk ID: {r['chunk_id']}, Distance: {r['distance']:.4f}, Text: {r['text'][:100]}...")
    if os.path.exists(dummy_file_path_minilm):
        os.remove(dummy_file_path_minilm)

    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main_run_standalone())