"""中转（ops/train_relay.py）在连不上、长等待、释放机器时的行为：见 tests/relay_sim.py。"""

import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class TestRelay(unittest.TestCase):
    def test_unreachable_heartbeat_release(self):
        # 模拟会改写 time 和 subprocess，放在单独的进程里跑
        p = subprocess.run([sys.executable, os.path.join(HERE, "relay_sim.py")], capture_output=True, text=True,
                           timeout=300)
        self.assertIn("ALL OK", p.stdout, p.stdout[-2000:] + p.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
