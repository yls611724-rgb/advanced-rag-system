
from cache.redis_cache import RedisCache


cache = RedisCache(ttl=60)

question = "Transformer 的核心机制是什么？"

fake_result = {
    "question": question,
    "answer": "Transformer 的核心机制是 Self-Attention。",
    "sources": [
        {
            "source": "AI基础.txt",
            "page": None,
            "score": 0.99
        }
    ]
}


# 1. 保存缓存
success = cache.set(question, fake_result)

print("缓存写入结果：", success)


# 2. 读取缓存
result = cache.get(question)

print("缓存读取结果：", result)


# 3. 检查结果
print(
    "答案是否一致：",
    result["answer"] == fake_result["answer"]
)


# 4. 删除缓存
deleted = cache.delete(question)

print("缓存删除结果：", deleted)


# 5. 再次读取
result_after_delete = cache.get(question)

print("删除后的缓存：", result_after_delete)
