"""CLI 启动器。保留这一层，是为了让 `python main.py` 这个用法继续可用。

真正的实现在 ragqa/cli.py，这里只做转发，不含任何逻辑。
"""
from ragqa.cli import main

if __name__ == "__main__":
    main()
