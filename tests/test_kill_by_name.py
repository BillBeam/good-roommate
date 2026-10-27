"""按名字结束进程的命令不能误杀自己（2026-10-26 17:10：pkill -f 匹配到了执行它的那条 ssh 命令本身）。"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().rsplit(")", 1)[1].split()[0] != "Z"


@unittest.skipUnless(os.path.isdir("/proc"), "需要 /proc（Linux）")
class TestKillByName(unittest.TestCase):
    def test_stale_vllm_killer_spares_its_caller(self):
        d = tempfile.mkdtemp()
        mod = "fake_vllm_srv_for_test"
        with open(os.path.join(d, mod + ".py"), "w") as f:
            f.write("import time\ntime.sleep(120)\n")
        srv = subprocess.Popen([sys.executable, "-m", mod], cwd=d, start_new_session=True)
        time.sleep(0.5)
        # 调用者自己的命令行里就写着这个模块名（就像 17:10 那条 ssh 命令里写着 netwatch.sh）
        code = (f"import sys; sys.path.insert(0, {ROOT!r}); from roommate.train import serve; "
                f"print('killed', serve.kill_stale({mod!r})); print('caller alive')  # -m {mod}")
        out = subprocess.run(["bash", "-c", f"{sys.executable} -c \"{code}\"; echo shell-alive-{mod}"],
                             capture_output=True, text=True, timeout=60)
        srv.wait(timeout=10)
        self.assertIn("caller alive", out.stdout)
        self.assertIn("shell-alive-", out.stdout)
        self.assertFalse(alive(srv.pid))
        shutil.rmtree(d)

    def test_netwatch_ctl_spares_its_caller(self):
        w = tempfile.mkdtemp()
        os.makedirs(os.path.join(w, "repo", "ops", "remote"))
        for f in ("netwatch.sh", "netwatch_ctl.sh"):
            shutil.copy(os.path.join(ROOT, "ops", "remote", f), os.path.join(w, "repo", "ops", "remote", f))
        env = dict(os.environ, NETWATCH_W=w)
        ctl = os.path.join(w, "repo", "ops", "remote", "netwatch_ctl.sh")
        r = subprocess.run(["bash", ctl, "start"], capture_output=True, text=True, env=env, timeout=30)
        self.assertIn("netwatch running", r.stdout)
        pid = int(open(os.path.join(w, "out", "diag", "netwatch.pid")).read())
        # 调用者的命令行里写着 netwatch.sh，用 restart / stop 都不能把调用者自己杀掉
        r = subprocess.run(["bash", "-c", f"bash {ctl} restart; echo still-here-netwatch.sh"], capture_output=True,
                           text=True, env=env, timeout=30)
        self.assertIn("still-here-netwatch.sh", r.stdout)
        self.assertIn("netwatch running", r.stdout)
        self.assertFalse(alive(pid))
        pid2 = int(open(os.path.join(w, "out", "diag", "netwatch.pid")).read())
        r = subprocess.run(["bash", "-c", f"bash {ctl} stop; echo still-here-netwatch.sh"], capture_output=True,
                           text=True, env=env, timeout=30)
        self.assertIn("still-here-netwatch.sh", r.stdout)
        self.assertIn("netwatch NOT running", r.stdout)
        time.sleep(0.5)
        self.assertFalse(alive(pid2))
        shutil.rmtree(w, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
