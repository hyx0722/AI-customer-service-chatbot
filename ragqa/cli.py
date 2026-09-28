"""CLI 入口。只负责组装依赖、跑交互循环、把答案打到 stdout。

检索/生成/记忆的实现在 ragqa/service.py；记忆规则在 ragqa/facts.py；
配置在 ragqa/config.py。这里不再有任何模块级副作用 ——
导入本模块不会连接数据库或加载模型。
"""
import logging

from ragqa.config import RAGConfig
from ragqa.service import RAGService


def _setup_logging() -> None:
    """根 logger 压到 WARNING，只放开本项目自己的进度日志。

    直接 basicConfig(level=INFO) 会把 httpx / huggingface_hub 的 INFO 也放出来，
    启动时刷出一整屏 HTTP Request 日志，把真正的进度信息淹掉。
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    # 只放开父 logger "ragqa"，子 logger（ragqa.service 等）自身是 NOTSET，
    # 有效级别沿父链继承，所以以后新增模块不必再来这里补名字。
    # "__main__" 单独列出是为了兼容 `python -m ragqa.cli` 这种运行方式。
    for name in ("ragqa", "__main__"):
        logging.getLogger(name).setLevel(logging.INFO)


log = logging.getLogger(__name__)


def main() -> None:
    _setup_logging()

    cfg = RAGConfig.from_env()
    service = RAGService(cfg)

    # 进度走 stderr（logging），对话内容走 stdout（print）。
    # 分开之后：管道里只拿到问答正文，而终端上进度与对话的先后顺序也不会错乱。
    log.info("会话 ID：%s   长期记忆命名空间：%s", cfg.session_id, cfg.memory_namespace)
    service.connect()
    log.info("正在建索引")
    service.build_index()
    log.info("就绪。我是一个问答助手，请输入你要询问的问题（quit / exit / q 退出）")

    turn = 1
    try:
        while True:
            print(f"=====第{turn}轮对话开始=====")
            question = input("> ").strip()
            if not question:
                continue
            if question.lower() in cfg.exit_words:
                print("对话结束")
                break
            print("====最终回答====")
            print(service.answer(question, cfg.session_id))
            print(f"=====第{turn}轮对话结束=====\n\n")
            turn += 1
    finally:
        service.close()


if __name__ == "__main__":
    main()
