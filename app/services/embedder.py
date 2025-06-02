import logging

class Embedder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 初始化向量化模型，这里假设使用 OpenAI 的嵌入模型
        self.embedding_model = "text-embedding-ada-002"
        self.length_func

    async def embed(self, chunks: list) -> None:
        """
        将文本块转换为向量
        :param chunks: 输入的文本块列表
        :return: None
        """
        if not chunks:
            return

        
        