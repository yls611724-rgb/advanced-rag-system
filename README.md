
# Advanced RAG 智能知识库问答系统

基于 LangChain 构建的 Advanced RAG（检索增强生成）智能知识库问答系统，支持 PDF、TXT 多格式文档加载，通过混合检索、RRF 融合、BGE Reranker 重排序等技术提高知识检索质量，并结合 DeepSeek 大语言模型生成答案。

项目使用 FastAPI 提供问答接口，使用 Redis 缓存问答结果，支持重复问题的快速响应。

## 一、项目介绍

传统大语言模型直接回答专业问题时，可能出现知识过时、回答不准确或产生幻觉等问题。

本项目通过 RAG 技术，将用户问题与本地知识库中的文档进行检索匹配，再将检索到的相关资料交给大语言模型生成答案，使回答能够参考指定知识库内容，并返回对应的参考来源。

## 二、核心功能

- **多格式文档加载**：支持 PDF、TXT 文档加载。
- **文本切分**：使用 RecursiveCharacterTextSplitter 将长文档切分为适合检索的文本块。
- **向量检索**：使用 BGE Embedding 将文本转换为向量，并存储在 FAISS 中。
- **关键词检索**：使用 BM25 算法进行关键词检索。
- **混合检索**：结合 FAISS 向量检索与 BM25 关键词检索。
- **RRF 排序融合**：使用 Reciprocal Rank Fusion 融合不同检索器的结果。
- **精排重排序**：使用 BGE Reranker 对候选文档进行相关性排序。
- **答案生成**：使用 DeepSeek 模型结合检索上下文生成答案。
- **相关性判断**：根据 Reranker 分数阈值决定是否生成答案。
- **Redis 缓存**：缓存问答结果，减少相同请求的重复计算。
- **RESTful API**：使用 FastAPI 提供问答接口。
- **接口测试**：使用 pytest 对 API 进行测试。

## 三、技术栈

| 技术 | 用途 |
|---|---|
| Python | 项目开发语言 |
| LangChain | RAG 流程编排 |
| DeepSeek | 大语言模型与答案生成 |
| BAAI/bge-small-zh-v1.5 | 文本向量化 |
| BAAI/bge-reranker-base | 文档相关性重排序 |
| FAISS | 向量数据库 |
| BM25 | 关键词检索 |
| RRF | 多路检索结果融合 |
| Redis | 问答结果缓存 |
| FastAPI | 后端 API 服务 |
| PyMuPDF | PDF 文档加载 |
| Docker | Redis 容器运行 |
| pytest | 自动化测试 |


## 四、项目架构设计

### 4.1 整体架构

本项目主要分为文档处理、检索、重排序、答案生成、缓存和 API 服务六个部分。

```mermaid
flowchart TD
    A["用户上传的知识库文档<br/>PDF / TXT"] --> B["文档加载<br/>Document Loader"]
    B --> C["文本切分<br/>Text Splitter"]
    C --> D["Document Chunks"]

    D --> E["BGE Embedding"]
    E --> F["FAISS 向量数据库"]

    D --> G["BM25 关键词检索器"]

    H["用户问题"] --> I["FastAPI 接口"]
    I --> J{"Redis 缓存查询"}

    J -->|"缓存命中"| K["直接返回历史答案"]
    J -->|"缓存未命中"| L["Hybrid Search"]

    F --> L
    G --> L

    L --> M["RRF 排序融合"]
    M --> N["候选文档"]
    N --> O["BGE Reranker"]
    O --> P{"相关性分数判断"}

    P -->|"达到阈值"| Q["Prompt + 检索上下文"]
    Q --> R["DeepSeek LLM"]
    R --> S["生成答案和来源"]

    P -->|"未达到阈值"| T["返回拒答信息"]

    S --> U["Redis 缓存结果"]
    T --> U
    U --> V["返回 API 响应"]

    K --> V



## 五、项目目录结构

```text

rag_project/
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── cache/
│   ├── __init__.py
│   └── redis_cache.py
│
├── chains/
│   ├── __init__.py
│   └── rag_chain.py
│
├── config/
│   ├── __init__.py
│   └── redis_client.py
│
├── loaders/
│   ├── __init__.py
│   ├── document_loader.py
│   └── document_splitter.py
│
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py
│   ├── bm25_retriever.py
│   ├── hybrid_search.py
│   └── reranker.py
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── scripts/
│   ├── __init__.py
│   ├── demo_advanced_rag.py
│   ├── demo_bm25.py
│   ├── demo_document_loader.py
│   ├── demo_document_splitter.py
│   ├── demo_hybrid_search.py
│   ├── demo_rag_components.py
│   ├── demo_redis_basic.py
│   ├── demo_redis_cache.py
│   ├── demo_reranker.py
│   └── demo_vector_store.py
│
├── evaluation/
│   ├── __init__.py
│   └── rag_test_dataset.json
│
├── data/
│   └── .gitkeep
│
├── build_multi_db.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
                   # 项目说明文档
```

### 5.1 核心模块说明

| 模块 | 主要职责 |
|---|---|
| api | 提供 FastAPI HTTP 接口 |
| loaders | 加载 PDF、TXT 文档并完成文本切分 |
| retrieval | 实现 FAISS、BM25、混合检索、RRF 和重排序 |
| chains |串联检索、精排、Prompt 和 DeepSeek 生成 |
| cache | 实现 Redis 问答结果缓存 |
| config | 管理 Redis 客户端连接 |
| scripts | 存放各模块的独立演示脚本 |
| tests | 对 API 和 Redis 功能进行测试 |
|evaluation | 存放检索评估数据集 |



## 六、环境配置与项目运行

### 6.1 环境要求

| 环境 | 版本或说明 |
|---|---|
| Python | 3.13.12（已验证环境） |
| Redis | 7.x |
| Docker | 用于运行 Redis |
| 操作系统 | Windows / Linux |
| DeepSeek API | 需要配置 API Key |

### 6.2 克隆项目

```bash
git clone <你的 GitHub 仓库地址>
cd rag_project
```

### 6.3 创建 Python 虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

### 6.4 安装项目依赖

```bash
python -m pip install -r requirements.txt
```

### 6.5 配置 DeepSeek API Key

复制项目提供的环境变量模板。

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

Linux / macOS：

```bash
cp .env.example .env
```

打开项目根目录下的 `.env`，填写自己的 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
```

请勿将真实 `.env` 文件或 API Key 上传到 GitHub。

### 6.6 启动 Redis

确保 Docker Desktop 已启动。

首次运行时，在终端执行：

```bash
docker run -d --name rag-redis -p 6379:6379 -v redis_data:/data redis:7-alpine redis-server --appendonly yes
```

检查 Redis：

```bash
docker exec rag-redis redis-cli ping
```

如果返回：

```text
PONG
```

说明 Redis 启动成功。

如果之前已经创建过 `rag-redis` 容器，但容器处于停止状态，则执行：

```bash
docker start rag-redis
```

### 6.7 准备知识库文档

将自己的 PDF、TXT 文件放入项目根目录的 `data/` 文件夹。

目录示例：

```text
rag_project/
└── data/
    ├── document1.pdf
    ├── document2.txt
    └── document3.pdf
```

注意：

- 支持 PDF 和 TXT 文件。
- TXT 文件使用 UTF-8 编码。
- 文档需要直接放在 `data/` 目录下，不要放在其子目录中。
- 原始文档默认不上传到 GitHub，需要自行准备。

### 6.8 构建 FAISS 向量数据库

确保当前终端位于项目根目录，并且已经准备好 `data/` 中的文档。

执行：

```bash
python build_multi_db.py
```

脚本将自动完成：

1. 加载 PDF、TXT 文档。
2. 使用 RecursiveCharacterTextSplitter 切分文本。
3. 使用 BAAI/bge-small-zh-v1.5 生成文本向量。
4. 使用 FAISS 创建向量数据库。
5. 将数据库保存到项目根目录的 `multi_vector_db/` 文件夹。
6. 执行一次示例相似度检索。

首次运行需要下载 Embedding 模型，请确保网络环境能够访问 Hugging Face，或提前准备好本地模型缓存。

注意：

- 首次构建完成后，会生成 `multi_vector_db/`。
- 更换知识库文档后，需要重新执行构建脚本。
- 当前版本采用全量重建方式，不支持增量更新。
- 不要将生成的 FAISS 数据库上传到 GitHub。

### 6.9 启动 FastAPI 服务

确认以下条件均已满足：

- 已配置 DeepSeek API Key。
- Redis 服务正常运行。
- `data/` 中已准备知识库文档。
- 已成功生成 `multi_vector_db/`。

在项目根目录执行：

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

服务启动后，可以通过浏览器访问：

```text
http://127.0.0.1:8000/
```

健康检查：

```text
http://127.0.0.1:8000/health
```

### 6.10 访问 API 交互文档

浏览器打开：

```text
http://127.0.0.1:8000/docs
```

可以在 Swagger UI 中查看接口、填写请求参数并测试 RAG 问答功能。

## 七、API 使用示例

### 7.1 请求接口

项目通过 FastAPI 提供 RAG 问答接口。

**请求地址：**

```http
POST http://127.0.0.1:8000/rag
```

**请求参数：**

```json
{
    "question": "Transformer 的核心机制是什么？",
    "candidate_k": 10,
    "top_k": 3
}
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| question | string | 必填 | 用户提出的问题 |
| candidate_k | int | 10 | 混合检索阶段的候选文档数量 |
| top_k | int | 3 | Reranker 最终保留的文档数量 |

### 7.2 响应结果

接口返回 JSON 格式的数据，包含用户问题、模型生成的答案以及参考文档来源。

示例：

```json
{
    "question": "Transformer 的核心机制是什么？",
    "answer": "Transformer 的核心机制是 Self-Attention（自注意力机制）。它能够计算序列中不同位置之间的关联关系，从而捕获上下文信息。",
    "sources": [
        {
            "source": "AI基础.txt",
            "page": null,
            "score": 0.999144
        },
        {
            "source": "Transformer.pdf",
            "page": 106,
            "score": 0.006231
        },
        {
            "source": "Transformer.pdf",
            "page": 15,
            "score": 0.002769
        }
    ]
}
```

注：以上为响应格式示例，具体答案和检索分数会随问题、文档内容及模型运行情况变化。Reranker 分数是相关性排序分数，不代表答案正确率或概率。
注：`sources` 中的 `page` 使用从 0 开始的页码索引。例如，`page: 0` 表示 PDF 第 1 页；TXT 文件没有页码时，返回 `null`。`score` 为 Reranker 输出的相关性排序分数，不代表答案正确率或概率。
### 7.3 接口测试

启动 FastAPI 后，访问：

```text
http://127.0.0.1:8000/docs
```

在 Swagger 页面中找到 `POST /rag`，点击 `Try it out`，填写请求参数并执行，即可测试问答功能。

### 7.4 参数校验

项目使用 Pydantic 对请求参数进行校验：

- question 不能为空。
- candidate_k 必须在 1～50 之间。
- top_k 必须在 1～10 之间。

不符合参数要求时，FastAPI 会返回对应的参数校验错误。


## 八、测试结果与功能验证

### 8.1 FastAPI 接口测试

使用 pytest 对 FastAPI 接口进行自动化测试。

运行命令：

```bash
pytest tests/test_api.py -v
```

测试结果：

```text
test_home PASSED
test_health PASSED
test_rag_success PASSED
test_rag_invalid_parameter PASSED
test_rag_value_error PASSED
test_rag_internal_error PASSED

========================
6 passed, 1 warning
========================
```

测试覆盖内容：

| 测试项目 | 测试内容 | 结果 |
|---|---|---|
| 首页接口 | 检查 API 是否正常运行 | 通过 |
| 健康检查 | 检查 RAG 系统状态 | 通过 |
| 问答接口 | 测试正常问答请求 | 通过 |
| 参数校验 | 测试非法参数 | 通过 |
| 异常处理 | 测试业务参数异常 | 通过 |
| 内部异常 | 测试服务内部异常处理 | 通过 |

注：接口自动化测试使用模拟 RAG 对象，通过 Mock 方式验证 API 路由、参数校验及异常处理，不会在每个测试用例中真实调用 Embedding、Reranker 和 DeepSeek 大语言模型。

本次测试结果为 6 passed, 1 warning。警告来自 LangChain Community 相关组件的弃用提示，不影响本次测试通过。
### 8.2 Redis 缓存测试

通过 Redis Python 客户端验证缓存读写、读取和删除功能。

测试结果：

```text
缓存写入结果： True

缓存读取结果：
{
    'question': 'Transformer 的核心机制是什么？',
    'answer': 'Transformer 的核心机制是 Self-Attention。',
    'sources': [...]
}

答案是否一致： True

缓存删除结果： True
删除后的缓存： None
```

### 8.3 RAG 缓存命中测试

通过连续提交相同问题，验证 Redis 缓存是否正常工作。

第一次请求：

```text
Redis 缓存未命中！开始执行 RAG 流程。
Hybrid 检索完成，候选文档数量：10
Reranker 重排序完成，最终文档数量：3
RAG 结果已尝试写入 Redis 缓存！
```

第二次请求：

```text
Redis 缓存命中！直接返回缓存结果。
```

测试结果表明：相同问题及相同检索参数再次请求时，可以直接读取缓存结果，避免重复执行检索、重排序和大模型生成。
缓存机制说明：

- 缓存 Key 由用户问题、candidate_k 和 top_k 共同决定，并使用 SHA-256 生成哈希值。
- 相同问题及相同检索参数再次请求时，可以命中已有缓存。
- 缓存默认 TTL 为 3600 秒，即 1 小时。
- 缓存保存完整问答结果，包括问题、答案和来源信息。
- 当前缓存 Key 不包含知识库版本，知识库更新后需要进一步完善缓存失效机制。
### 8.4 项目功能验证总结

目前已完成以下功能验证：

- PDF、TXT 多格式文档加载。
- 文档清洗与文本切分。
- BGE Embedding 向量化及 FAISS 向量检索。
- BM25 关键词检索。
- FAISS + BM25 混合检索及 RRF 排序融合。
- BGE Reranker 文档重排序。
- DeepSeek 大语言模型答案生成。
- 基于相关性分数阈值的拒答机制。
- Redis 问答缓存。
- FastAPI 问答接口。
- pytest 接口自动化测试。


## 九、项目总结与后续优化方向

### 9.1 项目总结

本项目基于 LangChain 构建了一个 Advanced RAG 智能知识库问答系统，实现了从多格式文档加载、文本切分、混合检索、结果融合、文档重排序到大语言模型答案生成的完整流程。

在基础 RAG 的基础上，进一步引入 BM25、RRF、BGE Reranker 等技术，优化知识库检索流程，并使用 Redis 缓存重复问答结果。

通过 FastAPI 将 RAG 核心流程封装为 HTTP 服务，并使用 pytest 完成接口自动化测试。

通过本项目，实践了以下技术：

- LangChain 与大语言模型应用开发。
- Embedding 模型与向量数据库。
- BM25、Hybrid Search、RRF 等检索技术。
- Cross-Encoder Reranker 重排序。
- Redis 缓存设计与应用。
- FastAPI 后端接口开发。
- 使用 Docker 运行 Redis 服务。
- pytest 自动化测试。

### 9.2 当前项目的不足

目前项目已经实现基本的 Advanced RAG 问答流程，但仍有一些可以改进的地方：

1. **检索效果评估**：目前已有基础检索指标实验，但评估数据规模较小，后续需要构建更完整的测试集。
2. **知识库更新机制**：目前主要读取预先构建的 FAISS 数据库，尚未实现完整的增量文档更新流程。
3. **缓存一致性**：当前缓存采用 TTL 过期机制，知识库更新时还需要考虑缓存失效和版本管理。
4. **服务部署**：目前已完成本地 FastAPI 服务和 Docker Redis 运行，后续可以进一步使用 Docker Compose 完成整体服务部署。
5. **系统可观测性**：后续可以增加日志、请求耗时统计和异常监控。

### 9.3 后续优化方向

- 增加知识库增量更新与删除功能。
- 完善检索效果评估与测试集。
- 引入 Docker Compose 实现服务一键启动。
- 增加日志记录和请求耗时监控。
- 优化 RAG Prompt 和上下文管理。
- 进一步探索 Agent、工具调用与 LangGraph 工作流。
