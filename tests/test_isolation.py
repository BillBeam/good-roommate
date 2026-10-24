"""训练与环境代码不接触外部基准数据。"""

import builtins
import os
import unittest
from unittest import mock

from roommate.world import generate_world

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IsolationTest(unittest.TestCase):
    def test_world_generation_reads_no_files(self):
        def deny(*a, **k):
            raise AssertionError(f"生成世界时不应读写文件：{a[:1]}")
        with mock.patch.object(builtins, "open", deny):
            generate_world(0)

    def test_env_code_never_references_benchmarks(self):
        # 环境与（今后的）训练会用到的模块；report.py 只负责写报告，其中提到 external/ 是说明文字
        pkg = os.path.join(ROOT, "roommate")
        for fn in os.listdir(pkg):
            if fn.endswith(".py") and fn != "report.py":
                with open(os.path.join(pkg, fn), encoding="utf-8") as f:
                    src = f.read()
                for bad in ("external", "ATRBench", "import bench", "from bench", "huggingface"):
                    self.assertFalse(bad in src, f"{fn} 引用了 {bad}")

    def test_external_dir_ignored(self):
        with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as f:
            lines = f.read().split()
        self.assertIn("external/", lines)


if __name__ == "__main__":
    unittest.main()
