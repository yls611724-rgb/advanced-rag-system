from config.redis_client import get_redis_client


# 1. 获取 Redis 客户端
client = get_redis_client()

# 2. 保存一条数据
client.set(
    "username",
    "zhangsan"
)
print("数据保存成功！")


# 3. 读取数据
value = client.get("username")
print("读取到的数据：", value)

# 4. 设置过期时间：60秒
client.expire("username", 60)

# 5. 查看剩余过期时间
ttl = client.ttl("username")
print("剩余过期时间：", ttl, "秒")

# 6. 设置一个测试数据，2秒后过期
client.set("test_expire", "hello", ex=2)
print("过期前：", client.get("test_expire"))
import time
time.sleep(3)
print("过期后：", client.get("test_expire"))