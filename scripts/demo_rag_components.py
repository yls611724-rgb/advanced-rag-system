
from chains.rag_chain import (
    get_llm,
    get_rag_prompt,
    format_context
)

# 测试 DeepSeek 初始化
llm = get_llm()
print("DeepSeek 模型初始化成功！")

# 测试 Prompt
prompt = get_rag_prompt()
print("RAG Prompt 创建成功！")

# 测试上下文整理
print("\n========== 测试上下文整理 ==========")

from langchain_core.documents import Document

test_results = [
    (
        0.9991,
        Document(
            page_content="Transformer 是一种基于注意力机制的神经网络架构。",
            metadata={
                "source": "AI基础.txt"
            }
        )
    ),
    (
        0.0062,
        Document(
            page_content="Self-Attention 是 Transformer 的核心组成部分。",
            metadata={
                "source": "Transformer.pdf",
                "page": 10
            }
        )
    )
]

context = format_context(test_results)

print(context)
