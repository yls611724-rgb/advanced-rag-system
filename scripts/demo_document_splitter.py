
from loaders.document_loader import load_documents
from loaders.document_splitter import split_documents


# 第一步：加载原始文档
documents = load_documents()

# 第二步：切分文档
chunks = split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
)

# 第三步：查看切分结果
print("\n========== Chunk 测试 ==========")

for i, chunk in enumerate(chunks[:5]):

    print(f"\n第 {i + 1} 个 Chunk")

    print("内容：", chunk.page_content[:150])

    print("内容长度：", len(chunk.page_content))

    print("元数据：", chunk.metadata)
