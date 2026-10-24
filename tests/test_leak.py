"""选项不泄题：不看用户的规则选对率不超过 1/6 + 5 个百分点；正确答案的位置均匀。"""

import unittest

from roommate import probes
from roommate.world import generate_world

LIMIT = 1 / 6 + 0.05


class LeakTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = [generate_world(s) for s in (1, 2, 3)]

    def test_modal_rule(self):
        r = sum(probes.modal_rule(w) for w in self.gate) / 3
        self.assertLessEqual(r, LIMIT)

    def test_conservative_rule(self):
        r = sum(probes.conservative_rule(w) for w in self.gate) / 3
        self.assertLessEqual(r, LIMIT)

    def test_position(self):
        from collections import Counter
        cnt = Counter()
        for w in self.gate:
            cnt += probes.position_rule(w)[1]
        n = sum(cnt.values())
        self.assertLessEqual(max(cnt.values()) / n, LIMIT)
        x, p = probes.chi2_uniform(cnt)
        self.assertGreater(p, 0.01)

    def test_classifier_small(self):
        train = [generate_world(s) for s in range(100, 110)]
        r = probes.classifier_probe(train, self.gate)
        self.assertLessEqual(r["test_acc"], LIMIT)

    def test_chi2_sf(self):
        # 已知值：自由度 5 时 11.07 对应上尾 0.05
        self.assertAlmostEqual(probes.chi2_sf(11.0705, 5), 0.05, places=3)
        self.assertAlmostEqual(probes.chi2_sf(9.4877, 4), 0.05, places=3)


if __name__ == "__main__":
    unittest.main()
