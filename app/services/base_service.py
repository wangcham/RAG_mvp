# 封装异步操作
import asyncio
import logging
from services.database import SessionLocal # 导入 SessionLocal 工厂函数

class BaseService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_session_factory = SessionLocal # 使用 SessionLocal 工厂函数

    async def _run_sync(self, func, *args, **kwargs):
        """
        在单独的线程中运行同步函数。
        如果第一个参数是 session，则在 to_thread 中获取新的 session。
        """
        # 如果函数需要数据库会话作为第一个参数，我们在这里获取它
        if getattr(func, '__name__', '').startswith('_db_'): # 约定：数据库操作的同步方法以 _db_ 开头
            session = await asyncio.to_thread(self.db_session_factory)
            try:
                result = await asyncio.to_thread(func, session, *args, **kwargs)
                return result
            finally:
                session.close()
        else:
            # 否则，直接运行同步函数
            return await asyncio.to_thread(func, *args, **kwargs)