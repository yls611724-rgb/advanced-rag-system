import json
import hashlib
import logging
import redis
from config.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, ttl=3600):
        """
        ttl：缓存过期时间，单位为秒，默认1小时
        """
        self.client = get_redis_client()
        self.ttl = ttl

    def _make_key(self, question, candidate_k=10, top_k=3):
        """
        根据问题和检索参数生成唯一缓存键
        """
        content = f"{question.strip()}:{candidate_k}:{top_k}"
        hash_value = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
        return f"rag:answer:{hash_value}"

    def get(self, question, candidate_k=10, top_k=3):
        """
        从 Redis 获取缓存
        """
        key = self._make_key(question, candidate_k, top_k)
        try:
            value = self.client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.warning("Redis 缓存读取失败：%s", e)
            return None

    def set(self, question, result, candidate_k=10, top_k=3):
        """
        将 RAG 完整结果保存到 Redis
        """
        key = self._make_key(question, candidate_k, top_k)
        try:
            value = json.dumps(
                result,
                ensure_ascii=False
            )
            self.client.set(
                key,
                value,
                ex=self.ttl
            )
            return True
        except (redis.RedisError, TypeError, ValueError) as e:
            logger.warning("Redis 缓存写入失败：%s", e)
            return False

    def delete(self, question, candidate_k=10, top_k=3):
        """
        删除指定问题的缓存
        """
        key = self._make_key(question, candidate_k, top_k)
        try:
            self.client.delete(key)
            return True
        except redis.RedisError as e:
            logger.warning("Redis 缓存删除失败：%s", e)
            return False
