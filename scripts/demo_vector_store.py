
from loaders.document_loader import load_documents
from loaders.document_splitter import split_documents

from retrieval.vector_store import (
    get_embeddings,
    build_vector_store,
    load_vector_store,
    search_documents
)


# ==================================================
# 1. 加载文档
# ==================================================

documents = load_documents()


# ==================================================
# 2. 文本切分
# ==================================================

chunks = split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
)


# ==================================================
# 3. 加载 Embedding 模型
# ==================================================

embeddings = get_embeddings()


# ==================================================
# 4. 创建 FAISS 数据库并保存
# ==================================================

vector_db = build_vector_store(
    documents=chunks,
    embeddings=embeddings,
    save_path="multi_vector_db"
)


# ==================================================
# 5. 重新加载数据库
# ==================================================

vector_db = load_vector_store(
    embeddings=embeddings,
    db_path="multi_vector_db"
)


# ==================================================
# 6. 测试检索
# ==================================================

query = "Transformer 的核心机制是什么？"

results = search_documents(
    vector_db=vector_db,
    query=query,
    k=3
)


# ==================================================
# 7. 输出结果
# ==================================================

print("\n========== FAISS 检索结果 ==========")

for i, doc in enumerate(results):

    print(f"\n第 {i + 1} 个检索结果")

    print("内容：", doc.page_content[:300])

    print("来源：", doc.metadata)
