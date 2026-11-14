
def doc_key(doc):
    """
    为 Document 生成唯一标识。
    内容、来源、页码都相同，才认为是同一个 Chunk。
    """
    return (
        doc.page_content,
        doc.metadata.get("source"),
        doc.metadata.get("page")
    )


def reciprocal_rank_fusion(result_lists, k=60):
    """
    使用 RRF 融合多个检索结果列表。

    result_lists：多个检索器返回的 Document 列表
    k：RRF 平滑参数，通常使用 60
    """
    fused_scores = {}
    documents = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc_key(doc)

            if key not in fused_scores:
                fused_scores[key] = 0.0
                documents[key] = doc

            fused_scores[key] += 1 / (k + rank)

    ranked_keys = sorted(
        fused_scores,
        key=lambda key: fused_scores[key],
        reverse=True
    )

    return [
        (documents[key], fused_scores[key])
        for key in ranked_keys
    ]


class HybridRetriever:
    """
    BM25 + FAISS + RRF 混合检索器。
    """

    def __init__(self, vector_db, bm25_retriever):
        self.vector_db = vector_db
        self.bm25_retriever = bm25_retriever

    def search(self, query, k=5, fetch_k=5):
        if not query or not query.strip():
            raise ValueError("检索问题不能为空！")

        if k < 1:
            raise ValueError("k 必须大于等于 1！")

        if fetch_k < 1:
            raise ValueError("fetch_k 必须大于等于 1！")

        # 1. FAISS 语义检索
        vector_results = self.vector_db.similarity_search(
            query,
            k=fetch_k
        )

        # 2. BM25 关键词检索
        bm25_results = self.bm25_retriever.search(
            query,
            k=fetch_k
        )

        # 3. RRF 融合
        fused_results = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=60
        )

        # 4. 取融合后的 Top-K
        return [
            doc for doc, score in fused_results[:k]
        ]
