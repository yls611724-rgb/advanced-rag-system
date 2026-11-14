from pathlib import Path

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# 1. 加载所有文档
# =========================

data_dir = Path("data")
all_documents = []

# 加载 PDF
for pdf_file in data_dir.glob("*.pdf"):
    loader = PyMuPDFLoader(str(pdf_file))
    documents = loader.load()
    all_documents.extend(documents)

# 加载 TXT
for txt_file in data_dir.glob("*.txt"):
    loader = TextLoader(
        str(txt_file),
        encoding="utf-8"
    )
    documents = loader.load()
    all_documents.extend(documents)

print("Document 总数量：", len(all_documents))

# =========================
# 2. 文本切分
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(all_documents)
print("切分后的 Chunk 数量：", len(chunks))

# =========================
# 3. 加载 Embedding 模型
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
# =========================
# 4. 创建 FAISS 向量数据库
# =========================

vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
print("FAISS 向量数据库创建成功！")

# =========================
# 5. 保存数据库
# =========================

vector_db.save_local("multi_vector_db")
print("多文档向量数据库保存成功！")

# =========================
# 6. 测试检索
# =========================

query = "Transformer 的核心机制是什么？"
results = vector_db.similarity_search(
    query,
    k=3
)
for i, doc in enumerate(results):
    print(f"\n===== 检索结果 {i + 1} =====")
    print("内容：", doc.page_content[:300])
    print("来源：", doc.metadata)
