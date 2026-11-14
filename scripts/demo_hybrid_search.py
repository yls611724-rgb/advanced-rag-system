
from pathlib import Path

from loaders.document_loader import load_documents
from loaders.document_splitter import split_documents

from retrieval.bm25_retriever import LocalBM25Retriever
from retrieval.vector_store import get_embeddings, load_vector_store
from retrieval.hybrid_search import HybridRetriever


# 第一步：加载并切分文档
documents = load_documents()
chunks = split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
)

# 第二步：创建 BM25 检索器
bm25_retriever = LocalBM25Retriever(chunks)

# 第三步：加载已经保存的 FAISS 数据库
embeddings = get_embeddings()

project_root = Path(__file__).resolve().parent.parent
vector_db_path = project_root / "multi_vector_db"

vector_db = load_vector_store(
    embeddings,
    db_path=vector_db_path
)

# 第四步：创建混合检索器
hybrid_retriever = HybridRetriever(
    vector_db=vector_db,
    bm25_retriever=bm25_retriever
)

# 第五步：测试检索
query = "Transformer 的核心机制是什么？"

results = hybrid_retriever.search(
    query=query,
    k=5,
    fetch_k=5
)

print("\n========== Hybrid 混合检索结果 ==========")
print("问题：", query)
print("最终返回数量：", len(results))

for i, doc in enumerate(results, start=1):
    print(f"\n第 {i} 个检索结果")
    print("内容：", doc.page_content[:300])
    print("来源：", doc.metadata)
