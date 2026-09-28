"""长期记忆的单元测试：抽取规则 + 存储读写 + 解耦守卫。

运行（在项目根目录下）：python -m pytest tests/ -v

这些测试不连接 Milvus / Postgres、不加载嵌入模型、不调用大模型。
ragqa/facts.py 刻意不导入任何重依赖，所以整套测试是秒级的 ——
重构前因为 import main 会拉起嵌入模型，同样这批用例要跑 22 秒。
"""
import subprocess
import sys
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore

from ragqa.facts import (
    extract_facts,
    format_memories_block,
    load_memories,
    remember_facts,
)

# 本文件在 tests/ 下，上一级即项目根。守卫测试起子进程时要用它当 cwd。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def as_dict(text):
    """把 [(slug, fact), ...] 转成 dict 方便断言。"""
    return dict(extract_facts(text))


# ---------------- 姓名 ----------------

@pytest.mark.parametrize("text, expected", [
    ("我叫张三", "用户的名字是张三"),
    ("我的名字是李四", "用户的名字是李四"),
    # slug 归一到 name，但事实文本保留用户自己的说法（姓名/名字）
    ("我的姓名是周七", "用户的姓名是周七"),
    ("我名叫王五", "用户的名字是王五"),
    ("你好，我叫赵六，很高兴认识你", "用户的名字是赵六"),
])
def test_name(text, expected):
    """无论哪种说法，姓名都必须落在同一个 name slug 上。"""
    assert as_dict(text)["name"] == expected


def test_name_phrasings_share_one_slug():
    """回归测试：曾出现「我叫张三」→ name 而「我的名字是李四」→ slot:名字，
    两条同时存在会让提示词里出现互相矛盾的名字。"""
    assert list(as_dict("我叫张三").keys()) == ["name"]
    assert list(as_dict("我的名字是李四").keys()) == ["name"]


def test_name_slug_is_stable_so_repeat_overwrites():
    """同一个人重复报名，slug 必须一致 —— 否则会堆积成多条。"""
    a = as_dict("我叫张三")
    b = as_dict("我叫张三")
    assert list(a.keys()) == list(b.keys()) == ["name"]


# ---------------- 显式「记住」 ----------------

@pytest.mark.parametrize("text, expected", [
    ("记住：我偏好中文回答", "用户要求记住：我偏好中文回答"),
    ("请记住 我的生日是 5 月 1 日", "用户要求记住：我的生日是 5 月 1 日"),
    ("记住，不要再给我推荐英文资料", "用户要求记住：不要再给我推荐英文资料"),
])
def test_explicit_remember(text, expected):
    assert list(as_dict(text).values()) == [expected]


def test_explicit_takes_priority_and_does_not_stack():
    """显式指令优先级最高：一句「记住」不该同时刷出姓名等多条。"""
    facts = extract_facts("记住：我叫张三")
    assert len(facts) == 1
    assert facts[0][0].startswith("explicit:")


def test_explicit_different_content_does_not_overwrite():
    """两条不同的「记住」必须共存，不能被彼此的 slug 覆盖。"""
    a = list(as_dict("记住：我喜欢蓝色").keys())[0]
    b = list(as_dict("记住：我不吃香菜").keys())[0]
    assert a != b


# ---------------- 「我的 X 是 Y」 ----------------

@pytest.mark.parametrize("text, slug, expected", [
    ("我的邮箱是 a@b.com", "contact:email", "用户的邮箱是a@b.com"),
    ("我的电话是 13800138000", "contact:phone", "用户的电话是13800138000"),
    ("我的手机号是 13900139000", "contact:phone", "用户的手机号是13900139000"),
    ("我的工号是 A1234", "slot:工号", "用户的工号是A1234"),   # 未登记的同义表 → 保持 slot: 前缀
])
def test_my_slot(text, slug, expected):
    assert as_dict(text)[slug] == expected


def test_email_is_not_truncated_at_dot():
    """回归测试：取值段曾排除 ASCII 句点，导致 a@b.com 被截成 a@b。"""
    assert as_dict("我的邮箱是 a@b.com")["contact:email"].endswith("a@b.com")


def test_different_slots_do_not_overwrite():
    d = as_dict("我的邮箱是 a@b.com，我的电话是 13800138000")
    assert d["contact:email"] == "用户的邮箱是a@b.com"
    assert d["contact:phone"] == "用户的电话是13800138000"


def test_synonym_slots_normalize_to_one_slug():
    """「我的邮箱」和「我的邮件」是同一个槽位，不该存成两条。"""
    assert list(as_dict("我的邮箱是 a@b.com").keys()) == ["contact:email"]
    assert list(as_dict("我的邮件是 a@b.com").keys()) == ["contact:email"]


# ---------------- 偏好 ----------------

def test_preference_recorded():
    facts = as_dict("我喜欢简洁的回答")
    assert len(facts) == 1
    assert list(facts.values())[0] == "用户偏好简洁的回答"


# ---------------- 不该命中的反例 ----------------

@pytest.mark.parametrize("text", [
    "",
    "   ",
    "今天天气怎么样?",
    "豆包支持哪些功能?",
    "如何用 Python 写一个爬虫?",
    "我的问题是这个功能怎么用",      # 「我的问题」在屏蔽名单里，是泛指不是属性
    "我的理解是这样",                # 同上
    "请问这个多少钱?",
])
def test_negative_cases(text):
    assert extract_facts(text) == []


def test_blocklisted_slot_is_skipped():
    assert extract_facts("我的问题是怎么退款") == []
    assert extract_facts("我的建议是再等等") == []


# ---------------- 提问不得被当成陈述（真实踩到的 bug） ----------------

@pytest.mark.parametrize("question", [
    "我叫什么?",
    "我叫什么",
    "我的名字是什么?",
    "我的邮箱是什么?",
    "我叫谁?",
    "我的电话是多少?",
])
def test_questions_do_not_produce_facts(question):
    """回归测试：线上真实故障 —— 问「我叫什么?」时 _RE_NAME 匹配到
    「我叫」+「什么」，把已保存的姓名覆盖成了垃圾值「什么」。"""
    assert extract_facts(question) == []


def test_asking_the_name_does_not_clobber_the_stored_name():
    """先陈述后提问，提问不得改变原有事实（slug 必须仍是 name，值仍是张三）。"""
    assert as_dict("我叫张三") == {"name": "用户的名字是张三"}
    assert extract_facts("我叫什么?") == []


def test_explicit_remember_still_works_for_question_like_text():
    """显式「记住」优先于问句判断：用户明确要求记住的，即使是疑问句也照存。"""
    facts = extract_facts("记住：我不喜欢被问年龄")
    assert len(facts) == 1
    assert facts[0][1] == "用户要求记住：我不喜欢被问年龄"


# ---------------- 存储读写（用 InMemoryStore，不碰 Postgres） ----------------

NS = ("memories", "test-session")


@pytest.fixture
def store():
    return InMemoryStore()


def test_remember_then_load_roundtrip(store):
    saved = remember_facts(store, NS, "我叫张三")
    assert saved == [("name", "用户的名字是张三")]
    assert load_memories(store, NS) == ["用户的名字是张三"]


def test_restating_overwrites_instead_of_accumulating(store):
    """同一事实说三遍，只应留一条 —— slug 确定性就是为了这个。"""
    for _ in range(3):
        remember_facts(store, NS, "我叫张三")
    assert len(load_memories(store, NS)) == 1


def test_no_facts_means_no_writes(store):
    assert remember_facts(store, NS, "今天天气怎么样?") == []
    assert load_memories(store, NS) == []


def test_namespaces_are_isolated(store):
    """记忆按会话隔离：另一个 session 不该看到本会话的事实。"""
    other = ("memories", "other-session")
    remember_facts(store, NS, "我叫张三")
    remember_facts(store, other, "我叫李四")
    assert load_memories(store, NS) == ["用户的名字是张三"]
    assert load_memories(store, other) == ["用户的名字是李四"]


def test_multiple_facts_from_one_sentence(store):
    remember_facts(store, NS, "我的邮箱是 a@b.com，我的电话是 13800138000")
    texts = load_memories(store, NS)
    assert len(texts) == 2
    assert any("a@b.com" in t for t in texts)
    assert any("13800138000" in t for t in texts)


# ---------------- format_memories_block ----------------

def test_format_empty_returns_empty_string():
    """没有记忆时不能留下抬头，否则每轮提示词里都挂着一段空标题。"""
    assert format_memories_block([]) == ""


def test_format_contains_header_and_each_fact():
    block = format_memories_block(["用户的名字是张三", "用户偏好简洁的回答"])
    assert "已知的用户信息" in block
    assert "- 用户的名字是张三" in block
    assert "- 用户偏好简洁的回答" in block


def test_format_end_to_end_from_store(store):
    remember_facts(store, NS, "我叫张三")
    block = format_memories_block(load_memories(store, NS))
    assert "张三" in block


# ---------------- 解耦守卫 ----------------

def test_facts_module_pulls_in_no_heavy_dependencies():
    """ragqa/facts.py 必须保持轻量。

    这是「记忆规则已从模块级副作用里抽出」的可执行证明：
    一旦有人在 ragqa/facts.py 里加了 langchain / torch 之类的导入，
    导入代价会从毫秒级回到十几秒，这条测试会立刻失败。

    注意它同时守着 ragqa/__init__.py 必须为空：导入 ragqa.facts 会先执行父包的
    __init__.py，所以只要有人在那里加了 `from ragqa.service import RAGService`
    之类的便利导出，langchain 全家桶就会被拉进来，本测试立刻失败。
    """
    code = (
        "import sys; import ragqa.facts; "
        "heavy = [m for m in ('torch','sentence_transformers','transformers',"
        "'langchain','langchain_core','langgraph','pymilvus','psycopg',"
        "'psycopg_pool') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
        # conftest.py 插入的 sys.path 不会传给子进程（pytest 不设置 PYTHONPATH），
        # 只能靠 cwd 兜底：`-c` 会把 cwd 放进 sys.path[0]。
        cwd=str(PROJECT_ROOT),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"ragqa/facts.py 引入了重依赖: {out.stdout.strip()}"
