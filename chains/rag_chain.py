
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from cache.redis_cache import RedisCache

# 1. 加载 .env 文件
load_dotenv()


# 2. 初始化 DeepSeek
def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件！")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0
    )

    return llm


# 3. 创建 RAG Prompt
def get_rag_prompt():
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
你是一个专业的知识库问答助手。

请严格遵守以下规则：

1. 回答必须以提供的参考资料为依据。
2. 不允许编造参考资料中没有的信息。
3. 如果参考资料无法回答问题，请明确回答：
   根据当前知识库中的资料，无法回答这个问题。
4. 回答要清晰、准确、通俗易懂。
5. 如果使用了参考资料，请尽可能说明资料来源。
"""
        ),
        (
            "human",
            """
以下是从知识库中检索到的参考资料：

{context}

用户的问题是：

{question}

请根据参考资料回答。
"""
        )
    ])

    return prompt


# 4. 整理检索结果，构建上下文
def format_context(reranked_results):
    context_parts = []

    for rank, (score, doc) in enumerate(
        reranked_results,
        start=1
    ):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page")

        source_info = source

        if page is not None:
            # PyMuPDFLoader 的 page 从 0 开始
            source_info += f"（PDF 第 {page + 1} 页）"

        context_parts.append(
            f"""
[参考资料 {rank}]
来源：{source_info}
相关性分数：{score:.4f}
内容：
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)

class AdvancedRAGChain:
    """
    完整的 Advanced RAG 问答链。

    流程：
    Hybrid 检索 -> Reranker 重排序 -> Prompt -> DeepSeek
    """

    def __init__(
            self,
            hybrid_retriever,
            reranker,
            llm=None,
            min_relevance_score=0.01,
            cache=None
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm = llm if llm is not None else get_llm()
        self.prompt = get_rag_prompt()

        self.min_relevance_score = min_relevance_score
        self.cache = cache or RedisCache(ttl=3600)

        self.chain = (
                self.prompt
                | self.llm
                | StrOutputParser()
        )

    def invoke(self, question, candidate_k=10, top_k=3):
        """
        输入问题，返回最终答案和参考来源。
        优先查询 Redis 缓存，未命中时执行完整 RAG 流程。
        """

        # 1. 检查问题
        if not question or not question.strip():
            raise ValueError("用户问题不能为空！")

        if candidate_k < 1 or top_k < 1:
            raise ValueError("candidate_k 和 top_k 必须大于等于 1！")

        # 2. 查询 Redis 缓存
        cached_result = self.cache.get(
            question=question,
            candidate_k=candidate_k,
            top_k=top_k
        )

        if cached_result is not None:
            print("\nRedis 缓存命中！直接返回缓存结果。")
            return cached_result

        print("\nRedis 缓存未命中！开始执行 RAG 流程。")

        # 3. Hybrid 混合检索
        candidate_docs = self.hybrid_retriever.search(
            query=question,
            k=candidate_k,
            fetch_k=candidate_k
        )

        print(f"\nHybrid 检索完成，候选文档数量：{len(candidate_docs)}")

        # 4. Reranker 重排序
        reranked_results = self.reranker.rerank(
            query=question,
            documents=candidate_docs,
            top_k=top_k
        )

        print(f"Reranker 重排序完成，最终文档数量：{len(reranked_results)}")

        # 5. 检查是否有检索结果
        if not reranked_results:
            result = {
                "question": question,
                "answer": "根据当前知识库中的资料，无法回答这个问题。",
                "sources": []
            }

            # 缓存拒答结果
            self.cache.set(
                question=question,
                result=result,
                candidate_k=candidate_k,
                top_k=top_k
            )

            return result

        # 6. 检查最高相关性分数
        max_score = max(
            score for score, doc in reranked_results
        )

        # 7. 根据相关性阈值决定是否调用 DeepSeek
        if max_score < self.min_relevance_score:
            answer = "根据当前知识库中的资料，无法回答这个问题。"
        else:
            # 8. 整理参考资料
            context = format_context(reranked_results)

            # 9. 调用 DeepSeek 生成答案
            answer = self.chain.invoke({
                "context": context,
                "question": question
            })

        # 10. 整理来源信息
        sources = []
        for score, doc in reranked_results:
            source = {
                "source": doc.metadata.get("source", "未知来源"),
                "page": doc.metadata.get("page"),
                "score": float(score)
            }

            sources.append(source)

        # 11. 整理最终返回结果
        result = {
            "question": question,
            "answer": answer,
            "sources": sources
        }
        # 12. 保存完整结果到 Redis
        self.cache.set(
            question=question,
            result=result,
            candidate_k=candidate_k,
            top_k=top_k
        )
        print("RAG 结果已尝试写入 Redis 缓存！")
        return result
