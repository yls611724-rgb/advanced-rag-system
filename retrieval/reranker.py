
from sentence_transformers import CrossEncoder


class BGEReranker:
    """
    使用 BGE CrossEncoder 对候选文档进行重排序。
    """

    def __init__(
        self,
        model_name="BAAI/bge-reranker-base",
        max_length=512,
        device="cpu"
    ):
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device
        )

        print("BGE Reranker 模型加载成功！")

    def rerank(self, query, documents, top_k=3):
        """
        参数：
            query：用户问题
            documents：待重排序的 Document 列表
            top_k：最终返回多少个文档

        返回：
            [(相关性分数, Document), ...]
        """

        if not query or not query.strip():
            raise ValueError("检索问题不能为空！")

        if top_k < 1:
            raise ValueError("top_k 必须大于等于 1！")

        if not documents:
            return []

        # 1. 构造 Query-Document 配对
        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        # 2. 计算相关性分数
        scores = self.model.predict(pairs)

        # 3. 将分数和文档配对
        reranked_results = sorted(
            zip(scores, documents),
            key=lambda x: float(x[0]),
            reverse=True
        )

        # 4. 返回 Top-K
        return [
            (float(score), doc)
            for score, doc in reranked_results[:top_k]
        ]
