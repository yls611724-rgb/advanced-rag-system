
from pathlib import Path

from loaders.document_loader import load_documents
from loaders.document_splitter import split_documents

from retrieval.bm25_retriever import LocalBM25Retriever
from retrieval.vector_store import get_embeddings, load_vector_store
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import BGEReranker


# 1. 加载并切分文档
documents = load_documents()
chunks = split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
)

# 2. 创建 BM25
bm25_retriever = LocalBM25Retriever(chunks)

# 3. 加载 FAISS
embeddings = get_embeddings()

project_root = Path(__file__).resolve().parent.parent
vector_db_path = project_root / "multi_vector_db"

vector_db = load_vector_store(
    embeddings,
    db_path=vector_db_path
)

# 4. 创建 Hybrid 检索器
hybrid_retriever = HybridRetriever(
    vector_db=vector_db,
    bm25_retriever=bm25_retriever
)

# 5. 执行混合检索
query = "Transformer 的核心机制是什么？"

candidate_docs = hybrid_retriever.search(
    query=query,
    k=10,
    fetch_k=10
)

print("\n========== Hybrid 候选结果 ==========")
print("候选文档数量：", len(candidate_docs))

# 6. 加载 BGE Reranker
reranker = BGEReranker(
    model_name="BAAI/bge-reranker-base",
    max_length=512,
    device="cpu"
)

# 7. 执行重排序
reranked_results = reranker.rerank(
    query=query,
    documents=candidate_docs,
    top_k=5
)

# 8. 打印结果
print("\n========== Reranker 最终 Top 5 ==========")

for rank, (score, doc) in enumerate(
    reranked_results,
    start=1
):
    print(f"\n第 {rank} 名")
    print("相关性分数：", round(score, 4))
    print("内容：", doc.page_content[:200])
    print("来源：", doc.metadata)
