from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
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

from api.schemas import RAGRequest, RAGResponse
import logging

# 放在 main.py 文件顶部的 import 区域
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_rag_chain():
    """ 初始化完整 RAG 系统。 """
    # 1. 加载文档
    documents = load_documents()

    # 2. 文档切分
    chunks = split_documents(documents)

    # 3. 加载 Embedding
    embeddings = get_embeddings()

    # 4. 加载 FAISS
    db_path = PROJECT_ROOT / "multi_vector_db"

    vector_db = load_vector_store(
        embeddings=embeddings,
        db_path=db_path
    )

    # 5. 创建 BM25
    bm25_retriever = LocalBM25Retriever(chunks)

    # 6. 创建 Hybrid
    hybrid_retriever = HybridRetriever(
        vector_db=vector_db,
        bm25_retriever=bm25_retriever
    )

    # 7. 加载 Reranker
    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-base",
        max_length=512,
        device="cpu"
    )

    # 8. 创建完整 RAG Chain
    rag = AdvancedRAGChain(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker
    )
    return rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务启动时初始化模型和检索系统。
    """
    print("正在初始化 RAG 系统……")
    app.state.rag = build_rag_chain()
    print("RAG 系统初始化完成！")
    yield
    print("RAG 服务正在关闭……")

app = FastAPI(
    title="Advanced RAG Knowledge Base",
    description="基于 Hybrid Search、RRF、BGE Reranker 和 DeepSeek 的知识库问答系统",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def home():
    return {
        "message": "Advanced RAG API is running!"
    }

@app.get("/health")
def health_check(request: Request):
    rag = getattr(request.app.state, "rag", None)

    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 系统尚未初始化"
        )

    return {
        "status": "healthy",
        "rag_initialized": True,
        "message": "Advanced RAG 服务运行正常"
    }


@app.post("/rag", response_model=RAGResponse)
def ask_rag(request: RAGRequest, http_request: Request):
    # 获取已经初始化好的 RAG 对象
    rag = http_request.app.state.rag
    try:
        result = rag.invoke(
            question=request.question,
            candidate_k=request.candidate_k,
            top_k=request.top_k
        )
        return result
    except ValueError as e:
        # 用户输入或参数不符合业务要求
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception:
        # 记录真实错误，方便开发者排查
        logger.exception("RAG 问答接口发生内部错误")
        # 不把内部代码、密钥等错误细节暴露给用户
        raise HTTPException(
            status_code=500,
            detail="RAG 服务内部错误，请稍后重试。"
        )