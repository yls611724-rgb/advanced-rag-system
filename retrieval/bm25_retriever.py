
import re
import jieba

from rank_bm25 import BM25Okapi


class LocalBM25Retriever:
    """
    基于 BM25 的本地关键词检索器。
    """

    def __init__(self, documents):
        """
        初始化 BM25 检索器。

        参数：
            documents: 切分后的 Document 列表
        """

        if not documents:
            raise ValueError("Document 列表不能为空！")

        self.documents = documents

        # 对所有文档进行分词
        self.tokenized_corpus = [
            self.tokenize(doc.page_content)
            for doc in documents
        ]

        # 创建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print("BM25 检索器创建成功！")
        print("文档数量：", len(self.documents))

    @staticmethod
    def tokenize(text):
        """
        中文文本分词，并过滤无效字符。
        """

        words = jieba.cut(text)

        tokens = []

        for word in words:

            word = word.strip()

            if not word:
                continue

            if re.search(r"[\u4e00-\u9fa5a-zA-Z0-9]", word):
                tokens.append(word.lower())

        return tokens

    def search(self, query, k=5):
        """
        根据用户问题检索相关文档。

        参数：
            query: 用户问题
            k: 返回结果数量

        返回：
            results: Document 列表
        """

        if not query.strip():
            raise ValueError("检索问题不能为空！")

        if k < 1:
            raise ValueError("k 必须大于等于 1！")

        # 对用户问题分词
        tokenized_query = self.tokenize(query)

        # 计算每个文档的 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)

        # 按分数从高到低排序
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        # 取 TopK
        top_indices = ranked_indices[:k]

        results = [
            self.documents[i]
            for i in top_indices
        ]

        return results
