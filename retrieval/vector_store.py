
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认向量数据库保存路径
DEFAULT_VECTOR_DB_PATH = PROJECT_ROOT / "multi_vector_db"


# ==================================================
# 1. 初始化 Embedding 模型
# ==================================================

def get_embeddings():

    """
    加载 BGE 中文 Embedding 模型。

    返回：
        embeddings: Embedding 模型对象
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("Embedding 模型加载成功！")

    return embeddings


# ==================================================
# 2. 创建 FAISS 向量数据库
# ==================================================

def build_vector_store(
    documents,
    embeddings,
    save_path=None
):

    """
    根据 Document 列表创建 FAISS 向量数据库。

    参数：
        documents: 切分后的 Document 列表
        embeddings: Embedding 模型
        save_path: 数据库保存路径

    返回：
        vector_db: FAISS 向量数据库对象
    """

    if not documents:
        raise ValueError("传入的 Document 列表不能为空！")

    # 创建 FAISS 数据库
    vector_db = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    print("FAISS 向量数据库创建成功！")

    # 如果传入保存路径，就保存数据库
    if save_path is not None:

        save_vector_store(
            vector_db,
            save_path
        )

    return vector_db


# ==================================================
# 3. 保存 FAISS 向量数据库
# ==================================================

def save_vector_store(
    vector_db,
    save_path=DEFAULT_VECTOR_DB_PATH
):

    """
    保存 FAISS 向量数据库。

    参数：
        vector_db: FAISS 数据库对象
        save_path: 保存路径
    """

    save_path = Path(save_path)

    # 创建父目录
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_db.save_local(str(save_path))

    print(f"FAISS 数据库保存成功：{save_path}")


# ==================================================
# 4. 加载已有 FAISS 数据库
# ==================================================

def load_vector_store(
    embeddings,
    db_path=DEFAULT_VECTOR_DB_PATH
):

    """
    加载本地 FAISS 向量数据库。

    参数：
        embeddings: Embedding 模型
        db_path: 数据库所在路径

    返回：
        vector_db: FAISS 数据库对象
    """

    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(
            f"FAISS 数据库不存在：{db_path}"
        )

    vector_db = FAISS.load_local(
        str(db_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"FAISS 数据库加载成功：{db_path}")

    return vector_db


# ==================================================
# 5. 相似度检索
# ==================================================

def search_documents(
    vector_db,
    query,
    k=5
):

    """
    根据用户问题检索相关文档。

    参数：
        vector_db: FAISS 数据库
        query: 用户问题
        k: 返回文档数量

    返回：
        results: 检索到的 Document 列表
    """

    if not query.strip():
        raise ValueError("检索问题不能为空！")

    if k < 1:
        raise ValueError("k 必须大于等于 1！")

    results = vector_db.similarity_search(
        query,
        k=k
    )

    return results
