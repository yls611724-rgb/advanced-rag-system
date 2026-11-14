import redis

def get_redis_client():
    """
    创建 Redis 客户端连接
    """
    client = redis.Redis(
        host="127.0.0.1",
        port=6379,
        db=0,
        decode_responses=True
    )
    return client

if __name__ == "__main__":
    client = get_redis_client()
    try:
        result = client.ping()
        if result:
            print("Redis Python 连接成功！")
            print("Ping 测试结果：", result)
    except redis.ConnectionError as e:
        print("Redis 连接失败：", e)
