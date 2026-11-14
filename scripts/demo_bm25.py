
from loaders.document_loader import load_documents
from loaders.document_splitter import split_documents

from retrieval.bm25_retriever import LocalBM25Retriever


# 1. 加载文档
documents = load_documents()

# 2. 文本切分
chunks = split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
)

# 3. 创建 BM25 检索器
bm25_retriever = LocalBM25Retriever(chunks)

# 4. 执行检索
query = "Transformer 的核心机制是什么？"

results = bm25_retriever.search(
    query=query,
    k=5
)

# 5. 输出结果
print("\n========== BM25 检索结果 ==========")

for i, doc in enumerate(results):

    print(f"\n第 {i + 1} 个检索结果")

    print("内容：", doc.page_content[:300])

    print("来源：", doc.metadata)
