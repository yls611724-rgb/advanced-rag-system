
from loaders.document_loader import load_documents


# 调用文档加载函数
documents = load_documents()


# 查看加载结果
print("\n========== 测试加载结果 ==========")

print("Document 总数量：", len(documents))


# 查看前 5 个 Document
for i, doc in enumerate(documents[:5]):

    print(f"\n第 {i + 1} 个 Document")

    print("内容：", doc.page_content[:100])

    print("元数据：", doc.metadata)
