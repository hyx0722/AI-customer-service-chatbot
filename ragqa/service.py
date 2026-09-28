"""RAGService：把检索与生成从模块级副作用里解耦出来。

导入本模块**不会**连接任何外部服务。外部资源（Milvus、Postgres、嵌入模型）
全部由显式 connect() 建立、close() 释放，所以：

  * 单测记忆规则不需要付 19 秒的模型加载与数据库连接代价
    （见 ragqa/facts.py —— 它压根不导入本模块）
  * 未来的 FastAPI 等调用方可以自己掌控生命周期，而不是被迫依赖全局单例
"""
import logging

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pymilvus import MilvusClient

# config 必须保持模块级导入，不要改成函数内惰性导入：它在导入时设置 HF_ENDPOINT，
# 而 huggingface_hub 只在自身 import 时读取该变量（详见 ragqa/config.py 顶部注释）。
from ragqa import config as config_module
from ragqa import facts as facts_module
from ragqa.config import RAGConfig

log = logging.getLogger(__name__)


class RAGService:
    def __init__(self, cfg: RAGConfig):
        # 只存配置，不做任何 I/O
        self.cfg = cfg
        self.client: MilvusClient | None = None
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.pool: ConnectionPool | None = None
        self.checkpointer: PostgresSaver | None = None
        self.store: PostgresStore | None = None
        self._middleware = None

    # ---------------- 生命周期 ----------------

    @property
    def connected(self) -> bool:
        return self.client is not None

    def _require_connection(self):
        if not self.connected:
            raise RuntimeError("RAGService 尚未连接，请先调用 connect()")

    def connect(self) -> "RAGService":
        """建立全部外部资源。幂等：重复调用不会重复建连。"""
        if self.connected:
            return self

        log.info("初始化 Milvus 客户端")
        self.client = MilvusClient(self.cfg.milvus_url)
        # 切库必须在这里做，不能只在建索引时做：任何"只检索不建库"的调用方
        # 否则都会到默认库 default 里找集合，报 collection not found
        self._ensure_database()

        log.info("加载嵌入模型 %s", self.cfg.embed_model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.cfg.embed_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

        log.info("连接 Postgres（checkpointer + 长期记忆 store）")
        self.pool = ConnectionPool(
            self.cfg.postgres_url,
            min_size=1,
            max_size=10,
            max_idle=300.0,       # 小于服务端 idle_session_timeout
            max_lifetime=3600.0,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "prepare_threshold": 0,
            },
        )
        self.checkpointer = PostgresSaver(self.pool)
        self.checkpointer.setup()

        # 长期记忆存储：与 checkpointer 共用同一个连接池，但迁移表是分开的，必须各自 setup()
        # 注意：不传 index= —— 语义检索需要 pgvector 扩展，而事实量级只有几十条，
        # 用命名空间精确读取就够了
        self.store = PostgresStore(conn=self.pool)
        self.store.setup()

        log.info("就绪")
        return self

    def close(self) -> None:
        """释放连接池。不关会打印一堆 couldn't stop thread 警告。"""
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        self.client = None
        self.embeddings = None
        self.checkpointer = None
        self.store = None

    def __enter__(self) -> "RAGService":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------- 索引 ----------------

    def _ensure_database(self) -> None:
        if self.cfg.db_name not in self.client.list_databases():
            self.client.create_database(self.cfg.db_name)
        self.client.use_database(self.cfg.db_name)

    def ensure_collection(self) -> None:
        """重建集合（drop + create）。"""
        self._require_connection()
        cfg = self.cfg
        if self.client.has_collection(collection_name=cfg.collection_name):
            self.client.drop_collection(collection_name=cfg.collection_name)
        self.client.create_collection(
            collection_name=cfg.collection_name,
            dimension=cfg.embed_dim,
            metric_type="COSINE",
        )

    def build_index(self) -> int:
        """读知识库 → 分块 → 向量化 → 写入 Milvus。返回分块数。"""
        self._require_connection()
        cfg = self.cfg
        self.ensure_collection()

        documents = TextLoader(file_path=cfg.knowledge_file, encoding="utf-8").load()

        # 只按空行(段落)切分，且不做重叠：
        # 段落是天然语义单元，重叠会在相邻块之间制造重复内容（旧配置 200/50 会切出
        # 一个 51 字符、其中 50 字符是重复的废块）。chunk_size 设得比最长段落小，
        # 这样每段独立成块；由于 separators 里没有更细的分隔符，超长段落也不会被拆开。
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=0,
            separators=["\n\n"],
        )
        chunks = splitter.split_documents(documents)
        texts = [c.page_content for c in chunks]
        vectors = self.embeddings.embed_documents(texts)

        data = [
            {
                "id": i,
                "vector": vectors[i],
                "text": chunks[i].page_content,
                "source": cfg.knowledge_file,
                "chunk_id": i,
            }
            for i in range(len(chunks))
        ]
        # upsert 可重复执行，覆盖同 id 数据
        self.client.upsert(collection_name=cfg.collection_name, data=data)
        self.client.flush(collection_name=cfg.collection_name)
        log.info("索引完成：%d 个分块", len(chunks))
        return len(chunks)

    # ---------------- 检索 ----------------

    def retrieve(self, query: str, limit: int = 3) -> list:
        """向量检索 Top-K，丢弃低于阈值的命中，并还原原文顺序。"""
        self._require_connection()
        cfg = self.cfg
        query_vector = self.embeddings.embed_query(str(query))
        results = self.client.search(
            collection_name=cfg.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text", "chunk_id", "source"],
        )
        # 丢弃低于阈值的命中：否则无关问题也会凑出若干段上下文，诱导模型硬编答案
        hits = [h for h in results[0] if h["distance"] >= cfg.retrieval_min_score]
        # 按 chunk_id 还原原文顺序：Milvus 是按相似度返回的，顺序是乱的
        hits.sort(key=lambda h: h["entity"].get("chunk_id") or 0)
        return hits

    # ---------------- 记忆 ----------------

    def remember(self, query: str) -> list[tuple[str, str]]:
        """抽取并保存事实。失败只记日志 —— 记忆写失败不该影响本轮回答。"""
        self._require_connection()
        try:
            saved = facts_module.remember_facts(self.store, self.cfg.memory_namespace, query)
        except Exception as e:
            log.warning("长期记忆写入失败：%s: %s", type(e).__name__, e)
            return []
        for slug, fact in saved:
            log.info("长期记忆已保存 %s = %s", slug, fact)
        return saved

    def _memories_block(self) -> str:
        try:
            texts = facts_module.load_memories(self.store, self.cfg.memory_namespace)
        except Exception as e:
            log.warning("长期记忆读取失败：%s: %s", type(e).__name__, e)
            return ""
        return facts_module.format_memories_block(texts)

    # ---------------- 模型 / 中间件 ----------------

    def _build_summarizer(self):
        """构造摘要中间件；构造不出来就返回 None。

        trigger 必须显式传：SummarizationMiddleware 的 trigger 默认是 None，
        开箱状态永远不会触发摘要（源码里 `if not self._trigger_conditions: return False`）。

        优先用 [fraction, messages] 双条件 —— fraction 依赖模型上报 max_input_tokens，
        deepseek 没有这个档案，构造时会抛 ValueError（已实测），
        所以必须降级成纯消息数触发，否则整个应用起不来。
        """
        cfg = self.cfg
        try:
            summary_model = init_chat_model(
                model=cfg.summary_model_name,
                api_key=config_module.require_env("DEEPSEEK_API_KEY"),
                base_url=config_module.require_env("DEEPSEEK_BASE_URL"),
                temperature=0,
            )
        except Exception as e:
            log.warning("摘要模型不可用（%s: %s），本次不做历史压缩", type(e).__name__, e)
            return None

        keep = ("messages", cfg.summary_keep_messages)
        candidates = [
            [("fraction", 0.6), ("messages", cfg.summary_trigger_messages)],
            [("messages", cfg.summary_trigger_messages)],
        ]
        for trigger in candidates:
            try:
                mw = SummarizationMiddleware(
                    model=summary_model,
                    trigger=trigger,
                    keep=keep,
                    summary_prompt=cfg.summary_prompt,
                )
                log.info("摘要中间件就绪 trigger=%s keep=%s", trigger, keep)
                return mw
            except Exception as e:
                log.warning("摘要中间件降级（%s: %s）", type(e).__name__, e)
        log.warning("摘要中间件不可用，本次不做历史压缩")
        return None

    def _get_middleware(self):
        """中间件只构造一次：get_agent() 每轮都调，摘要中间件不能跟着重建。"""
        if self._middleware is None:
            mw = [self._prompt_middleware()]
            summarizer = self._build_summarizer()
            if summarizer is not None:
                mw.insert(0, summarizer)
            self._middleware = mw
        return self._middleware

    def _prompt_middleware(self):
        """把长期记忆拼进系统提示词 —— store 里那些「永久事实」唯一的注入点。"""
        base = self.cfg.base_system_prompt
        service = self

        @dynamic_prompt
        def prompt_with_memories(request):
            return base + service._memories_block()

        return prompt_with_memories

    def get_agent(self):
        """构造对话 agent。每次调用返回新实例，但中间件是复用的。"""
        self._require_connection()
        model = init_chat_model(**config_module.chat_model_kwargs())
        return create_agent(
            model=model,
            tools=[],
            # 这里不给 system_prompt：由 prompt_with_memories 中间件动态返回，
            # 因为要把长期记忆拼进去
            middleware=self._get_middleware(),
            checkpointer=self.checkpointer,
            store=self.store,
        )

    # ---------------- 生成 ----------------

    def answer(self, query: str, thread_id: str) -> str:
        """跑完整条链路并**返回**答案文本（不打印）。

        返回而不是打印，是为了让调用方能复用 —— FastAPI 直接拿这个字符串回给前端。
        """
        self._require_connection()
        agent = self.get_agent()
        # thread_id 必须在同一段对话的每一轮保持不变，checkpointer 才会带回历史
        run_config = {"configurable": {"thread_id": thread_id}}

        # 先从**原始提问**里抽取事实写进长期记忆。必须在拼接检索上下文之前做，
        # 否则知识库正文里的字句会被误判成用户陈述；也必须在 invoke 之前做，
        # 这样本轮的系统提示词就已经能看到刚记住的内容。
        self.remember(query)

        hits = self.retrieve(query, limit=5)
        if hits:
            context_blocks = [
                f"[片段{i} | chunk_id={h['entity'].get('chunk_id', 'unknown')}"
                f" | source={h['entity'].get('source', 'unknown')}]\n{h['entity']['text']}"
                for i, h in enumerate(hits, 1)
            ]
            context = "\n\n".join(context_blocks)
        else:
            # 没有任何检索命中。但这段对话可能已有历史，模型仍可依据上文回答
            # （例如"我刚才问的是什么?"），所以只有历史也为空时才短路、省掉这次调用。
            if not agent.get_state(run_config).values.get("messages", []):
                return "我不知道。（没有检索到足够相关的内容，也没有对话历史）"
            context = "（本次没有检索到相关知识库内容，请依据对话历史回答）"

        user_prompt = f"""
    问题:
    {query}
    上下文:
    {context}
    """
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_prompt}]},
            config=run_config,
        )
        return result["messages"][-1].content
