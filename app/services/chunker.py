from langchain import text_splitter
from typing import List
import logging

class Chunker:
    def __init__(self,method: str):
        if method is None:
            self.method = "RecursiveCharacterTextSplitter"
        else:
            self.method = method

        self.logger = logging.getLogger(__name__)
        

    async def chunk(self, text: str) -> list:
        """
        将文本分割成块
        :param text: 输入的文本
        :return: 分割后的文本块列表
        """
        if not text:
            print("No text provided for chunking.")
            return []

        # 使用 langchain 的 text_splitter 进行文本分割
        if self.method == "RecursiveCharacterTextSplitter":
            splitter = text_splitter.RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
        else:
            raise ValueError(f"Unsupported chunking method: {self.method}")
        chunks = splitter.split_text(text)
        if not chunks:
            await self.logger.error("No chunks were created from the provided text.")
            return []
        return chunks