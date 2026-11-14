
from pathlib import Path

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader
)


def load_documents(data_dir=None):
    """
    加载指定目录下的 PDF 和 TXT 文档。

    参数：
        data_dir: 文档所在目录。
                  如果不传入，默认使用项目根目录下的 data 文件夹。

    返回：
        all_documents: LangChain Document 对象列表。
    """

    # 1. 确定 data 文件夹路径
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    # 2. 检查目录是否存在
    if not data_dir.exists():
        raise FileNotFoundError(
            f"文档目录不存在：{data_dir}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"指定路径不是文件夹：{data_dir}"
        )

    # 3. 创建列表，保存所有 Document
    all_documents = []

    # 4. 遍历目录下的所有文件
    for file_path in sorted(data_dir.iterdir()):

        # 5. 根据文件后缀判断文件类型
        suffix = file_path.suffix.lower()

        # 6. 加载 PDF
        if suffix == ".pdf":

            loader = PyMuPDFLoader(str(file_path))
            documents = loader.load()

            all_documents.extend(documents)

            print(
                f"PDF 加载成功：{file_path.name}，"
                f"页面数量：{len(documents)}"
            )

        # 7. 加载 TXT
        elif suffix == ".txt":

            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

            documents = loader.load()

            all_documents.extend(documents)

            print(
                f"TXT 加载成功：{file_path.name}，"
                f"Document 数量：{len(documents)}"
            )

    # 8. 返回所有 Document
    print("\n========== 文档加载完成 ==========")
    print("总 Document 数量：", len(all_documents))

    return all_documents
