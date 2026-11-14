
import pytest

from fastapi.testclient import TestClient

import api.main as main


# 模拟 RAG 系统，不加载真实模型
class FakeRAG:
    def invoke(self, question, candidate_k=10, top_k=3):

        if question == "触发业务异常":
            raise ValueError("测试用的参数错误")

        if question == "触发内部异常":
            raise RuntimeError("测试用的内部错误")

        return {
            "question": question,
            "answer": "这是自动化测试生成的模拟回答。",
            "sources": [
                {
                    "source": "AI基础.txt",
                    "page": None,
                    "score": 0.99
                }
            ]
        }


# 每个测试开始前，替换真实 RAG 初始化函数
@pytest.fixture
def client(monkeypatch):

    monkeypatch.setattr(
        main,
        "build_rag_chain",
        lambda: FakeRAG()
    )

    with TestClient(main.app) as test_client:
        yield test_client


# 测试 1：根路径
def test_home(client):

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Advanced RAG API is running!"


# 测试 2：健康检查
def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["rag_initialized"] is True


# 测试 3：正常问答
def test_rag_success(client):

    response = client.post(
        "/rag",
        json={
            "question": "Transformer 的核心机制是什么？",
            "candidate_k": 10,
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "Transformer 的核心机制是什么？"
    assert data["answer"] == "这是自动化测试生成的模拟回答。"
    assert len(data["sources"]) == 1


# 测试 4：参数校验失败
def test_rag_invalid_parameter(client):

    response = client.post(
        "/rag",
        json={
            "question": "",
            "candidate_k": 10,
            "top_k": 3
        }
    )

    assert response.status_code == 422


# 测试 5：业务异常
def test_rag_value_error(client):

    response = client.post(
        "/rag",
        json={
            "question": "触发业务异常",
            "candidate_k": 10,
            "top_k": 3
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "测试用的参数错误"


# 测试 6：内部异常
def test_rag_internal_error(client):

    response = client.post(
        "/rag",
        json={
            "question": "触发内部异常",
            "candidate_k": 10,
            "top_k": 3
        }
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "RAG 服务内部错误，请稍后重试。"
