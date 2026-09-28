# 智能客服问答机器人

基于 **Milvus 向量检索 + DeepSeek 生成 + Postgres 持久化** 的检索增强（RAG）问答 demo，带多轮对话历史和跨会话长期记忆。

命令行交互，问它知识库里的问题会依据检索到的原文回答，检索不到就明确说「我不知道」，而不是硬编一个答案。

## 功能特性

- **检索增强生成**：bge-m3 向量化 + Milvus 检索，带相似度阈值过滤（低于阈值的命中直接丢弃）
- **长期记忆**：纯正则抽取用户事实（姓名、邮箱、偏好、显式「记住：…」），**不调用大模型，零 API 成本**；slug 确定性生成，重复陈述是覆盖而非堆积
- **多轮对话**：LangGraph checkpointer 持久化到 Postgres，重启进程后同一会话仍能接上
- **历史自动压缩**：消息数超阈值时把旧轮对话摘成摘要，保留最近若干条
- **零副作用导入**：导入任何模块都不会连数据库或加载模型，因此单测是**秒级**的

## 架构

一问一答的完整链路（`ragqa/service.py` 的 `answer()`）：

```
用户提问
   │
   ├─1─ 从原始提问抽取事实 → 写入长期记忆（Postgres store）   ← 必须在检索之前
   │
   ├─2─ 向量检索知识库 Top-K → 丢弃低于阈值的命中 → 还原原文顺序
   │
   ├─3─ 拼成上下文（带 chunk_id / source 标注）
   │
   └─4─ 交给 agent 生成 ── 系统提示词 = 基础提示词 + 已记住的用户信息
                          （由 dynamic_prompt 中间件每轮动态拼接）
```

第 1 步必须在第 2 步之前：记忆要从**用户原始提问**里抽，不能从拼好检索上下文的 prompt 里抽，否则知识库正文里出现的「记住」「我的」等字样会被误判成用户自己的陈述。

### 分层设计

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `ragqa/config.py` | 读环境变量、构造配置对象 | 零副作用，可被任何模块安全导入；同时负责设置 HF 镜像 |
| `ragqa/facts.py` | 事实抽取与记忆读写 | **不导入任何重依赖**，store 按鸭子类型传入，可用 InMemoryStore 单测 |
| `ragqa/service.py` | 外部资源生命周期 + 检索 + 生成 | 导入时不连任何服务，全部由 `connect()` / `close()` 显式管理 |
| `ragqa/cli.py` | 组装依赖、跑交互循环 | 只打印，不含业务逻辑 |

依赖方向是单向的：`cli → service → facts`，且 `facts` 谁都不依赖。这条约束有可执行的测试守着（见下文「设计要点」）。

## 目录结构

```
.
├── ragqa/                  # 源码包
│   ├── __init__.py         # 刻意留空，勿加 re-export（见「设计要点」）
│   ├── config.py           # 配置读取 + 提示词
│   ├── facts.py            # 长期记忆：事实抽取与读写
│   ├── service.py          # RAGService：Milvus / Postgres / 嵌入模型 / agent
│   └── cli.py              # 交互式命令行
├── tests/
│   ├── conftest.py         # 把项目根加入 sys.path，使 import ragqa 免安装可用
│   └── test_facts.py       # 46 个用例：抽取规则 + 存储读写 + 解耦守卫
├── data/
│   └── knowledge.txt       # 知识库原文（按空行分段，每段一个语义单元）
├── main.py                 # 入口：python main.py
├── requirements.txt        # 本项目实际用到的依赖
└── requirements-full.txt   # 完整环境快照（含大量本项目未使用的包，备查）
```

## 快速开始

### 1. 前置依赖

- **Python 3.13**
- **Milvus**（默认 `localhost:19530`）—— 也可用 [Milvus Lite](https://milvus.io/docs/milvus_lite.md)
- **Postgres** —— 用于对话历史和长期记忆，首次运行会自动建表
- **DeepSeek API Key**

> 不需要 pgvector 扩展：事实量级只有几十条，用命名空间精确读取就够了，没建向量索引。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 .env

在项目根目录创建 `.env`：

```dotenv
# --- DeepSeek 模型 ---
DEEPSEEK_API_KEY="sk-你的密钥"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL_NAME_CHAT="deepseek-chat"

# --- Milvus ---
MILVUS_URL="http://localhost:19530"
DB_NAME="rag_tutorial"
COLLECTION_NAME="docs"

# --- 嵌入模型 ---
EMBED_MODEL_NAME="BAAI/bge-m3"
EMBED_DIM=1024

# --- 知识库与检索 ---
KNOWLEDGE_FILE="data/knowledge.txt"   # 相对项目根目录
RETRIEVAL_MIN_SCORE=0.55              # 相似度下限，低于此值的命中丢弃

# --- Postgres ---
POSTGRES_URL="postgresql://user:password@localhost:5432/langchain_db?sslmode=disable"

# --- 会话与记忆 ---
SESSION_ID="cli-default"              # 同时是对话线程 ID 和记忆命名空间
SUMMARY_TRIGGER_MESSAGES=12           # 消息数超过它就压缩历史
SUMMARY_KEEP_MESSAGES=6               # 压缩后保留最近几条
SUMMARY_MODEL_NAME="deepseek-chat"
```

**`RETRIEVAL_MIN_SCORE` 需要按你的知识库校准**：用 bge-m3 + COSINE 在本仓库自带知识库上实测，相关问题 top1 落在 0.66~0.79，无关问题落在 0.36~0.46，间隔约 0.20，所以取中点 0.55。换知识库或换嵌入模型后要重新测：取「最低的相关分数」和「最高的无关分数」的中点。

### 4. 运行

```bash
python main.py
```

启动时会重建向量索引（drop + create，幂等），然后进入交互循环。输入 `quit` / `exit` / `q` 退出。

进度日志走 **stderr**，对话内容走 **stdout**，所以 `python main.py > answer.txt` 只会拿到问答正文。

## 运行测试

```bash
python -m pytest tests/ -v
```

46 个用例，**亚秒级完成**，不连 Milvus / Postgres、不加载嵌入模型、不调用大模型。

## 设计要点

几个刻意的取舍，改代码前建议先读：

**`ragqa/__init__.py` 必须保持为空。** 导入 `ragqa.facts` 会先执行父包的 `__init__.py`，一旦在里面加了 `from ragqa.service import RAGService` 这类便利导出，langchain 全家桶就会被连带拉起，「记忆规则可以毫秒级导入」这条性质立刻退化。

**`tests/test_facts.py` 里有一条解耦守卫。** 它起子进程导入 `ragqa.facts`，断言 `sys.modules` 中没有 `torch` / `langchain` / `langgraph` / `pymilvus` / `psycopg` 等重依赖。这是「记忆规则已从模块级副作用里抽出」的可执行证明，不是形式主义 —— 加之前同样的用例要跑 22 秒，现在 0.16 秒。

**事实抽取用正则而非大模型。** 三层规则：显式「记住：X」优先级最高且不叠加其它规则；然后是姓名、「我的 X 是 Y」、偏好。槽位名做了同义归一（`名字`/`姓名` → `name`），否则「我叫张三」和「我的名字是李四」会存成两条互相矛盾的名字。

**提问不能被当成陈述。** 这是真实踩到过的线上故障：「我叫什么?」会被 `我叫` 规则匹配成「我叫」+「什么」，把已存的姓名覆盖成垃圾值。所以整句像问句、或捕获到的值本身是疑问词时都要拦掉。

**摘要中间件会优雅降级。** `SummarizationMiddleware` 的首选触发条件是 `[fraction, messages]` 双条件，但 `fraction` 依赖模型上报 `max_input_tokens`，deepseek 没有这个档案，构造时会抛 `ValueError`。代码按 `[双条件] → [纯消息数]` 顺序尝试，都失败就放弃压缩而不是让应用起不来。

**`HF_ENDPOINT` 镜像必须在 `HuggingFaceEmbeddings` 实例化之前设置。** `huggingface_hub` 只在自身 import 时读取该变量，所以 `config.py` 在模块顶层设置它，且 `service.py` 对 config 保持模块级导入 —— 别改成函数内惰性导入。

## 已知问题

- `config.py` 中 `exit_words` 含 `"quik"`，疑似 `"quick"` 的笔误（保留原样，改动会影响退出行为）。
- 无 Web 界面，仅命令行。`RAGService.answer()` 返回字符串而非打印，可直接被 FastAPI 之类的调用方复用。
