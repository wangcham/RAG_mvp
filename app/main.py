import asyncio
from quart import Quart,request
import sqlite_vss
from services.parser import FileParser
from services.chunker import Chunker
from services.retriever import Retriever
from services.embedder import Embedder
import os
import logging

import sqlite3
import sqlite_vss


class RAG_Manager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = FileParser()
        self.chunker = Chunker()
        self.retriever = Retriever()
        self.embedder = Embedder()
        self.db_path = "data/rag.db"

    async def retrieve_data(self):
        """检索流程"""
        pass

    async def store_data(self):
        """存储流程"""
        try:
            text = await self.parser.parse()
            chunks = await self.chunker.chunk(text)
            await self.embedder.embed(chunks)
        except Exception as e:
            await self.logger.error(f"Error in store_data: {str(e)}")
            return


if __name__ == "__main__":
    rag_manager = RAG_Manager()
    asyncio.run(rag_manager.store_data())














