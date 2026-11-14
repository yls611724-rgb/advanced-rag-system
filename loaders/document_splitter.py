
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
):
    """
    将 Document 列表切分成多个 Chunk。

    参数：
        documents: 原始 Document 列表
        chunk_size: 每个 Chunk 的目标字符长度
        chunk_overlap: 相邻 Chunk 的重叠字符长度

    返回：
        chunks: 切分后的 Document 列表
    """

    # 1. 过滤掉没有有效文本的 Document
    valid_documents = [
        doc
        for doc in documents
        if doc.page_content.strip()
    ]

    empty_count = len(documents) - len(valid_documents)

    print("原始 Document 数量：", len(documents))
    print("空白 Document 数量：", empty_count)
    print("有效 Document 数量：", len(valid_documents))

    # 2. 创建文本切分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # 3. 执行切分
    chunks = text_splitter.split_documents(valid_documents)

    print("\n========== 文本切分完成 ==========")
    print("Chunk 总数量：", len(chunks))

    return chunks
