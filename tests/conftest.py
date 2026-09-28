"""pytest 配置：把项目根加入 sys.path，让 `import ragqa` 无需安装即可解析。

为什么放在 tests/ 而不是项目根目录：
本项目没有 pytest.ini / pyproject.toml，rootdir 由命令行参数推断 ——
跑 `pytest`（不带参数）时 rootdir 是项目根，但跑 `pytest tests/` 时 rootdir
会变成 tests/，而 confcutdir 默认跟随 rootdir。放到项目根的 conftest.py
在后一种情况下**根本不会被加载**，`import ragqa` 就会失败。
放在这里，两种调用方式都能生效。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
