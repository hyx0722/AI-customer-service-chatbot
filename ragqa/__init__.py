"""智能客服 RAG 问答服务。

本文件必须保持为空 —— 除了这段文档字符串，不要在这里写任何 import 或 re-export。

原因：tests/test_facts.py::test_facts_module_pulls_in_no_heavy_dependencies
会起一个子进程导入 ragqa.facts，并断言 langchain / langgraph / pymilvus / torch
等重依赖没有出现在 sys.modules 里。而导入子模块**一定会先执行父包的
__init__.py**，所以这里一旦写下 `from ragqa.service import RAGService` 这类
便利导出，整个 langchain 全家桶就会被拉进来，那条守卫测试立刻失败 ——
而它的意义正是保证「记忆规则可以毫秒级导入」，这一点不能退化。
"""
