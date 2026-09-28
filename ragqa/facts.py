"""长期记忆：重要事实的抽取与读写。

本模块刻意**不导入** langchain / langgraph / pymilvus / torch 等重依赖。
store 通过参数传入并按鸭子类型使用（只需要 .put / .search / .delete），
所以可以用 langgraph 的 InMemoryStore 单测，完全不碰 Milvus 与 Postgres，
更不需要加载嵌入模型。

三层记忆里的前两层在这里（显式「记住：X」+ 规则匹配身份/偏好）。
全部是纯正则，不调用大模型，所以不增加任何 API 成本。
slug 是确定性的，因此同一事实重复陈述是覆盖同一条，而不是不断堆积。
"""
import hashlib
import re
from typing import Any, Protocol

_RE_EXPLICIT = re.compile(r"^\s*(?:请)?记住[：:，,\s]\s*(?P<v>.+?)[。.!！]?\s*$")
_RE_NAME = re.compile(r"我(?:叫|名字是|名叫)\s*(?P<v>[一-龥A-Za-z·]{1,20})")
# 取值段刻意**不**排除 ASCII 句点，否则邮箱、域名会被截断（a@b.com → a@b）。
# 中文句末的 。！？ 和逗号、分号仍然排除，避免把整句话都吃进来。
_RE_MY_SLOT = re.compile(
    r"我的(?P<slot>[一-龥A-Za-z]{1,10}?)(?:是|为)\s*(?P<v>[^\s，,。！!？?；;]{1,60})"
)
_RE_PREF = re.compile(r"我(?:喜欢|偏好|习惯)\s*(?P<v>[^\s，,。！!？?；;]{1,40})")

# 「我的X」里这些 X 不是可持久化的属性，只是对话中的泛指，不该写进长期记忆
_SLOT_BLOCKLIST = {
    "问题", "意思", "看法", "观点", "建议", "想法", "疑问", "理解",
    "答案", "需求", "目的", "理由", "方法", "方式", "印象", "感受",
}

# 同义槽位归一到同一个 slug。不做归一的话，「我叫张三」存成 name、
# 「我的名字是李四」存成 slot:名字，两条会同时存在，系统提示词里就出现
# 互相矛盾的两个名字。
_SLOT_ALIASES = {
    "名字": "name",
    "姓名": "name",
    "邮箱": "contact:email",
    "邮件": "contact:email",
    "电话": "contact:phone",
    "手机": "contact:phone",
    "手机号": "contact:phone",
}

# 疑问词。规则抽取最大的坑：提问会被当成陈述 ——
# 「我叫什么?」会被 _RE_NAME 匹配成「我叫」+「什么」，于是把已经记住的姓名
# 覆盖成垃圾值。必须在两个层面拦：整句像问句、以及捕获到的值本身是疑问词。
_INTERROGATIVES = ("什么", "啥", "谁", "多少", "怎么", "为什么", "吗", "呢", "哪一个")

# 注入提示词时的抬头
MEMORY_HEADER = "已知的用户信息（来自之前对话的长期记忆，可直接采信）："


class MemoryStore(Protocol):
    """只依赖真正用到的那两个方法，因此 InMemoryStore 和 PostgresStore 都能满足。"""

    def put(self, namespace: tuple[str, ...], key: str, value: dict) -> Any: ...
    def search(self, namespace_prefix: tuple[str, ...], /, *, limit: int = 10) -> list: ...


def _slug(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha1(value.encode('utf-8')).hexdigest()[:8]}"


def _is_question(text: str) -> bool:
    t = text.strip()
    if t.endswith(("?", "？")):
        return True
    return any(w in t for w in _INTERROGATIVES)


def _valid_value(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    return not any(w in v for w in _INTERROGATIVES)


def extract_facts(text: str) -> list[tuple[str, str]]:
    """从一条**原始用户提问**里抽取值得长期保存的事实，返回 [(slug, fact), ...]。

    纯函数、无副作用，便于单测。
    必须传原始提问，不能传拼好检索上下文的 prompt —— 否则知识库正文里出现的
    「记住」「我的」等字样会被误当成用户自己的陈述。
    """
    facts: list[tuple[str, str]] = []
    text = (text or "").strip()
    if not text:
        return facts

    # 1. 显式「记住：X」优先级最高：整句保留，且不再叠加其它规则。
    #    放在问句判断之前 —— 用户明确要求记住的内容，即使句式像疑问也照存。
    m = _RE_EXPLICIT.match(text)
    if m:
        value = m.group("v").strip()
        if value:
            # 用内容哈希做 key，保证不同的「记住」互不覆盖
            facts.append((_slug("explicit", value), f"用户要求记住：{value}"))
        return facts

    # 提问不是陈述事实。少了这一道，「我叫什么?」会把 name 覆盖成「什么」。
    if _is_question(text):
        return facts

    # 2. 姓名：固定 slug，重复说会覆盖
    m = _RE_NAME.search(text)
    if m and _valid_value(m.group("v")):
        facts.append(("name", f"用户的名字是{m.group('v').strip()}"))

    # 3. 「我的X是Y」：slug 带槽位名 → 「我的邮箱是」和「我的电话是」互不覆盖，
    #    但重复说「我的邮箱是」会覆盖；同义槽位（名字/姓名）归一到同一个 slug
    for m in _RE_MY_SLOT.finditer(text):
        slot = m.group("slot").strip()
        if slot in _SLOT_BLOCKLIST or not _valid_value(m.group("v")):
            continue
        slug = _SLOT_ALIASES.get(slot, f"slot:{slot}")
        facts.append((slug, f"用户的{slot}是{m.group('v').strip()}"))

    # 4. 偏好
    for m in _RE_PREF.finditer(text):
        if _valid_value(m.group("v")):
            facts.append((_slug("pref", m.group("v")), f"用户偏好{m.group('v').strip()}"))

    return facts


def remember_facts(
    store: MemoryStore,
    namespace: tuple[str, ...],
    query: str,
) -> list[tuple[str, str]]:
    """把提问里的事实写进长期记忆，返回成功写入的 [(slug, fact), ...]。

    写失败会让异常抛出，由调用方决定策略 ——
    service 层选择「记日志后继续回答」，因为记忆写失败不该影响本轮答案。
    """
    facts = extract_facts(query)
    for slug, fact in facts:
        store.put(namespace, slug, {"text": fact, "session": namespace[-1]})
    return facts


def load_memories(
    store: MemoryStore,
    namespace: tuple[str, ...],
    limit: int = 20,
) -> list[str]:
    """读回已保存的事实文本，按存储顺序返回。读失败让异常抛出。"""
    items = store.search(namespace, limit=limit)
    return [it.value.get("text", "") for it in items if it.value.get("text")]


def format_memories_block(texts: list[str]) -> str:
    """把事实列表拼成注入系统提示词的那一段。空列表返回空串（不加抬头）。"""
    if not texts:
        return ""
    lines = "\n".join(f"- {t}" for t in texts)
    return f"\n\n{MEMORY_HEADER}\n{lines}"
