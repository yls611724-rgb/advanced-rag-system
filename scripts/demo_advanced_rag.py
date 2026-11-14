
from pathlib import Path

from loaders.document_loader import load_documents
from loaders.document_splitter import split_documents

from retrieval.vector_store import (
    get_embeddings,
    load_vector_store
)
from retrieval.bm25_retriever import LocalBM25Retriever
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import BGEReranker

from chains.rag_chain import AdvancedRAGChain


# ==============================
# 1. 加载文档并切分
# ==============================

documents = load_documents()
chunks = split_documents(documents)

print("\nChunk 数量：", len(chunks))


# ==============================
# 2. 加载 Embedding 和 FAISS
# ==============================

embeddings = get_embeddings()

project_root = Path(__file__).resolve().parent.parent
db_path = project_root / "multi_vector_db"

vector_db = load_vector_store(
    embeddings=embeddings,
    db_path=db_path
)


# ==============================
# 3. 创建 BM25 和 Hybrid 检索器
# ==============================

bm25_retriever = LocalBM25Retriever(chunks)

hybrid_retriever = HybridRetriever(
    vector_db=vector_db,
    bm25_retriever=bm25_retriever
)


# ==============================
# 4. 加载 Reranker
# ==============================

reranker = BGEReranker(
    model_name="BAAI/bge-reranker-base",
    max_length=512,
    device="cpu"
)


# ==============================
# 5. 创建完整 RAG Chain
# ==============================

rag = AdvancedRAGChain(
    hybrid_retriever=hybrid_retriever,
    reranker=reranker
)

print("\nAdvanced RAG Chain 创建成功！")


# ==============================
# 6. 测试问答
# ==============================

question = "Transformer 的核心机制是什么？"

result = rag.invoke(
    question=question,
    candidate_k=10,
    top_k=3
)


# ==============================
# 7. 输出结果
# ==============================

print("\n========== 最终回答 ==========")
print(result["answer"])

print("\n========== 参考来源 ==========")

for source in result["sources"]:
    print(source)


# ==============================
# 8. 测试知识库外的问题
# ==============================

print("\n\n========== 测试知识库外问题 ==========")

outside_question = "中国的首都是哪里？"

outside_result = rag.invoke(
    question=outside_question,
    candidate_k=10,
    top_k=3
)

print("\n========== 知识库外问题的回答 ==========")
print(outside_result["answer"])

print("\n========== 检索到的参考来源 ==========")

for source in outside_result["sources"]:
    print(source)