"""菱形汇合、跳层边、双根、局部改权、负值与重挂的精确检查。"""
import copy
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from io import StringIO
from pathlib import Path

from warehouse_data import cli
from warehouse_data.facts import (
    FactSet,
    RebindResult,
    assess_rollup,
    allocate_fact,
    allocate_facts,
    build_warehouse_document,
    commit_warehouse,
    load_warehouse,
    plan_rebind,
    verify_warehouse,
    _document_digest,
)
from warehouse_data.hierarchy import (
    BridgePath,
    Edge,
    Hierarchy,
    format_weight,
    parse_weight,
)
from warehouse_data.io import load


def diamond_manifest():
    return {
        "kind": "weighted-hierarchy",
        "version": "d1",
        "nodes": ["s", "a", "b", "m", "r"],
        "edges": [
            {"child": "s", "parent": "a", "weight": "1/2"},
            {"child": "s", "parent": "b", "weight": "1/2"},
            {"child": "a", "parent": "m", "weight": "2/3"},
            {"child": "a", "parent": "r", "weight": "1/3"},
            {"child": "b", "parent": "m", "weight": "1/2"},
            {"child": "b", "parent": "r", "weight": "1/2"},
            {"child": "m", "parent": "r", "weight": "1"},
        ],
    }


def skip_manifest():
    return {
        "kind": "weighted-hierarchy",
        "version": "k1",
        "nodes": ["s", "a", "b", "r"],
        "edges": [
            {"child": "s", "parent": "a", "weight": "1"},
            {"child": "s", "parent": "r", "weight": "0"},
            {"child": "a", "parent": "b", "weight": "1/2"},
            {"child": "a", "parent": "r", "weight": "1/2"},
            {"child": "b", "parent": "r", "weight": "1"},
        ],
    }


class WeightTests(unittest.TestCase):
    def test_exact_rationals(self):
        self.assertEqual(parse_weight("3/4"), Fraction(3, 4))
        self.assertEqual(parse_weight("0"), Fraction(0))
        self.assertEqual(parse_weight("2"), Fraction(2))
        for bad in ("1.0", "0.5", "-1/2", "1/0", " 1/2", "1/2/3", 0.5, 1, True, None):
            with self.subTest(weight=bad):
                with self.assertRaises(ValueError):
                    parse_weight(bad)
        thirds = sum((Fraction(1, 3) for _ in range(3)), Fraction(0))
        self.assertEqual(thirds, 1)
        self.assertEqual(format_weight(Fraction(2, 4)), "1/2")
        self.assertEqual(format_weight(Fraction(7)), "7")


class ValidationTests(unittest.TestCase):
    def assert_invalid(self, mutate, message=None):
        manifest = diamond_manifest()
        mutate(manifest)
        with self.assertRaises(ValueError) as caught:
            Hierarchy.from_manifest(manifest)
        if message:
            self.assertIn(message, str(caught.exception))

    def test_shape_and_identity(self):
        self.assert_invalid(lambda v: v["nodes"].append("s"), "重复")
        self.assert_invalid(lambda v: v["nodes"].append(""), "身份")
        self.assert_invalid(lambda v: v.pop("nodes"), "节点")
        self.assert_invalid(lambda v: v["edges"].append({"child": "s", "parent": "ghost", "weight": "1"}))
        self.assert_invalid(lambda v: v["edges"].extend([
            {"child": "s", "parent": "a", "weight": "1/2"},
        ]), "重复边")
        self.assert_invalid(lambda v: v["edges"].append({"child": "r", "parent": "r", "weight": "1"}), "自环")
        self.assert_invalid(lambda v: v["edges"][0].update(weight="0.5"))
        self.assert_invalid(lambda v: v["edges"][0].update(weight="1/2 "))

    def test_weights_must_sum_to_exactly_one(self):
        def last_edge_bad(v):
            v["edges"].append({"child": "r", "parent": "s", "weight": "1"})
        self.assert_invalid(last_edge_bad, "环")
        self.assert_invalid(lambda v: v["edges"][2].update(weight="1/3"), "精确为1")
        self.assert_invalid(lambda v: v["edges"][0].update(weight="1/3"), "精确为1")

    def test_cycle_detected_even_with_zero_weight(self):
        cycle_manifest = {
            "version": "z1",
            "nodes": ["a", "b", "c"],
            "edges": [
                {"child": "a", "parent": "b", "weight": "0"},
                {"child": "a", "parent": "c", "weight": "1"},
                {"child": "b", "parent": "a", "weight": "1"},
                {"child": "b", "parent": "c", "weight": "0"},
            ],
        }
        # 每个非根节点权重和都是1，但 a<->b 环中的一条边是零权；
        # 零权不能成为绕过环检测的理由。
        with self.assertRaises(ValueError) as caught:
            Hierarchy.from_manifest(cycle_manifest)
        self.assertIn("环", str(caught.exception))

        # 权重和与环同时有问题时先报权重和，也是拒绝，不是接受。
        def broken_cycle(v):
            v["edges"].append({"child": "r", "parent": "r2", "weight": "0"})
            v["nodes"].append("r2")
            v["edges"].append({"child": "r2", "parent": "m", "weight": "0"})
        self.assert_invalid(broken_cycle)

    def test_zero_weight_edge_kept_in_structure(self):
        manifest = skip_manifest()
        hierarchy = Hierarchy.from_manifest(manifest)
        self.assertIn(Edge("s", "r", Fraction(0)), hierarchy.edges)
        weights = {(edge.child, edge.parent): edge.weight for edge in hierarchy.parents["s"]}
        self.assertEqual(weights, {("s", "a"): Fraction(1), ("s", "r"): Fraction(0)})
        self.assertEqual(format_weight(weights[("s", "r")]), "0")


class BridgeTests(unittest.TestCase):
    def bridge(self, version, source, ancestor):
        hierarchy = Hierarchy.from_manifest(version)
        return hierarchy.bridge_entry(source, ancestor)

    def test_identity_and_mult_root_conservation(self):
        hierarchy = Hierarchy.from_manifest(diamond_manifest())
        self.assertEqual(hierarchy.roots, ("r",))
        for node in hierarchy.nodes:
            self.assertEqual(hierarchy.rows[node][node].weight, 1)
            self.assertEqual(
                hierarchy.rows[node][node].paths,
                (BridgePath((node,), Fraction(1)),),
            )
            self.assertEqual(
                sum((entry.weight for entry in hierarchy.rows[node].values()
                     if entry.ancestor in hierarchy.roots), Fraction(0)),
                1,
            )
        for source, ancestor, expected in (
            ("s", "m", Fraction(7, 12)),
            ("s", "r", Fraction(1)),
            ("a", "r", Fraction(1)),
            ("b", "r", Fraction(1)),
        ):
            self.assertEqual(hierarchy.bridge_weight(source, ancestor), expected)
            self.assertIn(ancestor, hierarchy.rows[source])

    def test_diamond_multi_paths_are_added_not_split_or_shortest(self):
        hierarchy = Hierarchy.from_manifest(diamond_manifest())
        entry = hierarchy.bridge_entry("s", "m")
        # s 经 a 与经 b 两条路径分别 1/3 和 1/4，桥为 7/12；既不均分也不取单路。
        self.assertEqual(
            {path.nodes: path.weight for path in entry.paths},
            {
                ("s", "a", "m"): Fraction(1, 3),
                ("s", "b", "m"): Fraction(1, 2 * 2),
            },
        )
        self.assertEqual(sum((p.weight for p in entry.paths), Fraction(0)), Fraction(7, 12))
        root_entry = hierarchy.bridge_entry("s", "r")
        expected = {
            ("s", "a", "m", "r"): Fraction(1, 3),
            ("s", "b", "m", "r"): Fraction(1, 4),
            ("s", "a", "r"): Fraction(1, 6),
            ("s", "b", "r"): Fraction(1, 4),
        }
        self.assertEqual({path.nodes: path.weight for path in root_entry.paths}, expected)
        self.assertEqual(sum(expected.values(), Fraction(0)), 1)
        self.assertEqual(hierarchy.parent_counts()["s"], 2)
        self.assertEqual(hierarchy.parent_counts()["m"], 1)  # m 直接父只有 r；菱形在上层
        self.assertEqual(hierarchy.parent_edges("m")[0].parent, "r")

    def test_skip_level_and_zero_weight_path_evidence(self):
        hierarchy = Hierarchy.from_manifest(skip_manifest())
        entry = hierarchy.bridge_entry("s", "r")
        weights = {path.nodes: path.weight for path in entry.paths}
        self.assertEqual(
            weights,
            {
                ("s", "r"): Fraction(0),
                ("s", "a", "r"): Fraction(1, 2),
                ("s", "a", "b", "r"): Fraction(1, 2),
            },
        )
        self.assertEqual(entry.weight, 1)
        # 零权路径按节点序列排序不一定排在第一，但必须完整留在证据中。
        self.assertIn(0, [path.weight for path in entry.paths])
        self.assertEqual(sum(weights.values(), Fraction(0)), 1)

    def test_independent_enumeration_matches_dp_bridge(self):
        for manifest in (diamond_manifest(), skip_manifest()):
            hierarchy = Hierarchy.from_manifest(manifest)
            report = hierarchy.verify_bridge()
            self.assertTrue(report["ok"])
            self.assertTrue(report["independent_enumeration"])
            self.assertGreater(report["paths"], 0)
        report = Hierarchy.from_manifest(diamond_manifest()).verify_bridge()
        # s 到 m 与 r 多路径，路径证据多于桥条目；条目数=各节点可达祖先数。
        # s: s, a, b, m, r 五个条目；a: a, m, r 三个；b: b, m, r 三个；
        # m: m, r 两个；r: r 一个。路径数 s=6, a=3, b=3, m=2, r=1。
        self.assertEqual(report["entries"], 5 + 3 + 3 + 2 + 1)
        # 非自身路径：s 到 a/b/m/r 共 8 条（r 有 4 条），a 到 m/r 3 条，
        # b 到 m/r 3 条，m 到 r 1 条；再加 5 条自路径，共 20 条证据。
        self.assertEqual(report["paths"], 20)

    def test_many_parents_exact_thirds(self):
        hierarchy = Hierarchy.from_manifest({
            "version": "p1",
            "nodes": ["x", "p1", "p2", "p3"],
            "edges": [
                {"child": "x", "parent": "p1", "weight": "1/3"},
                {"child": "x", "parent": "p2", "weight": "1/3"},
                {"child": "x", "parent": "p3", "weight": "1/3"},
            ],
        })
        self.assertEqual(hierarchy.parent_counts()["x"], 3)
        self.assertEqual(set(hierarchy.roots), {"p1", "p2", "p3"})
        for root in hierarchy.roots:
            self.assertEqual(hierarchy.bridge_weight("x", root), Fraction(1, 3))

    def test_serialized_bridge_reloads_identically(self):
        from warehouse_data.facts import hierarchy_record
        for manifest in (diamond_manifest(), skip_manifest()):
            original = Hierarchy.from_manifest(manifest)
            reloaded = Hierarchy.from_mapping(hierarchy_record(original))
            self.assertEqual(original.digest, reloaded.digest)
            self.assertEqual(original.bridge_as_dict(), reloaded.bridge_as_dict())
            self.assertEqual(reloaded.verify_bridge()["ok"], True)


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.manifest, _ = load("data/normal.json")
        self.base = Hierarchy.from_manifest(self.manifest)
        self.facts = FactSet.from_manifest(self.manifest, hierarchy=self.base)

    def test_normal_allocations_paths_and_conservation(self):
        payload = allocate_facts(self.facts, self.base, include_paths=True)
        by_id = {row["fact_id"]: row for row in payload["facts"]}
        self.assertEqual(
            {row["ancestor"]: row["contribution"] for row in by_id["f1"]["contributions"]},
            {"leaf": "100", "left": "50", "right": "50", "rootA": "75", "rootB": "25"},
        )
        self.assertEqual(
            {row["ancestor"]: row["contribution"] for row in by_id["f2"]["contributions"]},
            {"leaf": "-20", "left": "-10", "right": "-10", "rootA": "-15", "rootB": "-5"},
        )
        self.assertIsNone(by_id["f3"]["value"])
        self.assertTrue(by_id["f3"]["missing"])
        for row in by_id["f3"]["contributions"]:
            self.assertIsNone(row["contribution"])  # null 不按 0 处理
        self.assertEqual(payload["measured_fact_count"], 2)
        self.assertEqual(payload["missing_fact_count"], 1)
        self.assertEqual(payload["conserved_measured_fact_count"], 2)
        self.assertEqual(by_id["f1"]["measured_root_total"], "100")
        self.assertIsNone(by_id["f3"]["measured_root_total"])
        # 路径证据：rootA 由 leaf->left->rootA 与 leaf->right->rootA 两路相加。
        root_a = {row["ancestor"]: row for row in by_id["f1"]["contributions"]}["rootA"]
        self.assertEqual(
            {tuple(path["nodes"]): path["weight"] for path in root_a["paths"]},
            {
                ("leaf", "left", "rootA"): "1/2",
                ("leaf", "right", "rootA"): "1/4",
            },
        )

    def test_null_and_negative_survive_fact_set_round_trip(self):
        rows = self.facts.as_dicts()
        self.assertIn({"fact_id": "f3", "node": "leaf", "value": None, "hierarchy_version": "hv1"}, rows)
        rebuilt = FactSet.from_rows(rows, nodes=self.base.nodes, known_versions={self.base.version})
        self.assertEqual(rebuilt.digest(), self.facts.digest())
        for bad in (0.0, 1.5, True, "100"):
            with self.assertRaises(ValueError):
                FactSet.from_rows([{"fact_id": "z", "node": "leaf", "value": bad,
                                    "hierarchy_version": "hv1"}])

    def test_version_mismatch_rejected(self):
        new = Hierarchy.successor(self.base, "hv2", self.manifest["new_edge_weights"])
        with self.assertRaises(ValueError):
            allocate_fact(self.facts.require("f1"), new)
        with self.assertRaises(ValueError):
            allocate_facts(self.facts, new)


class RollupTests(unittest.TestCase):
    def setUp(self):
        self.manifest, _ = load("data/normal.json")
        self.base = Hierarchy.from_manifest(self.manifest)
        self.facts = FactSet.from_manifest(self.manifest, hierarchy=self.base)

    def test_two_root_group_total(self):
        result = assess_rollup(self.facts, self.base, ["rootA", "rootB"])
        self.assertTrue(result["safe_to_add"])
        self.assertTrue(result["can_be_group_total"])
        self.assertEqual(result["group_total"], "80")
        self.assertEqual(result["candidate_unqualified_sum"], "80")
        self.assertEqual(result["fact_value_total"], "80")
        self.assertEqual(result["per_node"]["rootA"]["total"], "60")
        self.assertEqual(result["per_node"]["rootB"]["total"], "20")
        self.assertEqual(result["per_node"]["rootA"]["missing_fact_count"], 1)
        self.assertEqual(result["reasons"], [])
        self.assertNotIn("group_total_exact", result)

    def test_unrelated_root_can_be_included_without_changing_total(self):
        result = assess_rollup(self.facts, self.base, ["rootA", "rootB", "unrelated"])
        self.assertTrue(result["safe_to_add"])
        self.assertEqual(result["group_total"], "80")
        self.assertEqual(result["per_node"]["unrelated"]["total"], "0")
        self.assertEqual(result["per_node"]["unrelated"]["measured_fact_count"], 0)

    def test_ancestor_and_child_selection_not_additive(self):
        result = assess_rollup(self.facts, self.base, ["left", "rootA"])
        self.assertFalse(result["safe_to_add"])
        self.assertFalse(result["can_be_group_total"])
        self.assertIsNone(result["group_total"])
        self.assertEqual(result["candidate_unqualified_sum"], "100")
        self.assertEqual(result["fact_value_total"], "80")
        self.assertTrue(any(pair["ancestor"] == "rootA" and pair["descendant"] == "left"
                            for pair in result["ancestor_pairs"]))
        self.assertTrue(result["overlap_facts"])
        self.assertTrue(result["uncovered_facts"])
        self.assertTrue(any("祖先/后代" in reason for reason in result["reasons"]))
        self.assertTrue(any("共同下游" in reason for reason in result["reasons"]))

    def test_partial_root_set_not_group_total(self):
        result = assess_rollup(self.facts, self.base, ["rootA"])
        self.assertFalse(result["safe_to_add"])
        self.assertIsNone(result["group_total"])
        self.assertTrue(result["uncovered_facts"])
        self.assertEqual(result["per_node"]["rootA"]["total"], "60")

    def test_middle_siblings_are_not_roots_and_not_group_total(self):
        result = assess_rollup(self.facts, self.base, ["left", "right"])
        # 同层、无祖先关系、合并覆盖两分支（1/2+1/2=1），但它们是中间层：
        # 其贡献已构成根贡献的一部分，不能作为集团总量输出。
        self.assertFalse(result["root_membership"])
        self.assertFalse(result["can_be_group_total"])
        self.assertIsNone(result["group_total"])
        self.assertEqual(result["candidate_unqualified_sum"], "80")
        self.assertEqual(result["fact_value_total"], "80")
        self.assertTrue(any("非根" in reason for reason in result["reasons"]))

    def test_duplicate_selection_rejected(self):
        with self.assertRaises(ValueError):
            assess_rollup(self.facts, self.base, ["rootA", "rootA"])


class VersionAndRebindTests(unittest.TestCase):
    def setUp(self):
        self.manifest, _ = load("data/normal.json")
        self.base = Hierarchy.from_manifest(self.manifest)
        self.facts = FactSet.from_manifest(self.manifest, hierarchy=self.base)
        self.new = Hierarchy.successor(self.base, "hv2", self.manifest["new_edge_weights"])

    def test_local_reweight_recomputes_only_reverse_descendant_closure(self):
        self.assertEqual(self.new.parent_version, "hv1")
        self.assertEqual(self.new.changed_children, ("right",))
        self.assertEqual(self.new.recomputed_nodes, ("leaf", "right"))
        self.assertEqual(self.new.reused_nodes, ("left", "rootA", "rootB", "unrelated"))
        self.assertNotIn("left", self.new.recomputed_nodes)
        self.assertEqual(self.new.rows["left"]["rootA"].computed_in, "hv1")
        self.assertEqual(self.new.rows["right"]["rootA"].computed_in, "hv2")
        self.assertEqual(self.new.rows["leaf"]["rootA"].computed_in, "hv2")
        # 桥变化：right 1/2,1/2 -> 1/4,3/4；leaf 3/4,1/4 -> 5/8,3/8。
        self.assertEqual(self.new.bridge_weight("right", "rootA"), Fraction(1, 4))
        self.assertEqual(self.new.bridge_weight("right", "rootB"), Fraction(3, 4))
        self.assertEqual(self.new.bridge_weight("leaf", "rootA"), Fraction(5, 8))
        self.assertEqual(self.new.bridge_weight("leaf", "rootB"), Fraction(3, 8))
        self.assertEqual(self.new.rows["left"]["left"].weight, 1)
        self.assertEqual(self.new.verify_bridge()["ok"], True)
        for node in self.new.nodes:
            self.assertEqual(
                sum((entry.weight for entry in self.new.rows[node].values()
                     if entry.ancestor in self.new.roots), Fraction(0)),
                1,
            )

    def test_successor_validates_updates(self):
        with self.assertRaises(ValueError):
            Hierarchy.successor(self.base, "hv1", self.manifest["new_edge_weights"])
        with self.assertRaises(ValueError):
            Hierarchy.successor(self.base, "hv2", [])
        # 不能新增边，也不能只把一侧改小而使权重和不为1：
        # 单边改 1/4 会留下 1/4+1/2=3/4，必须显式报错。
        with self.assertRaises(ValueError):
            Hierarchy.successor(self.base, "hv2", [
                {"child": "right", "parent": "rootA", "weight": "1/4"},
            ])
        with self.assertRaises(ValueError):
            Hierarchy.successor(self.base, "hv2", [
                {"child": "right", "parent": "rootA", "weight": "1/4"},
                {"child": "right", "parent": "rootB", "weight": "3/4"},
                {"child": "ghost", "parent": "rootA", "weight": "1"},
            ])
        with self.assertRaises(ValueError):
            Hierarchy.successor(self.base, "hv2", [
                {"child": "right", "parent": "rootA", "weight": "1/4"},
                {"child": "right", "parent": "rootA", "weight": "1"},
            ])
        with self.assertRaises(ValueError) as caught:
            Hierarchy.successor(self.base, "hv2", [
                {"child": "leaf", "parent": "rootA", "weight": "1"},
            ])  # 未知边（leaf->rootA 不存在）
        self.assertIn("未知边", str(caught.exception))

    def test_deep_reverse_closure_on_diamond(self):
        diamond = Hierarchy.from_manifest(diamond_manifest())
        new = Hierarchy.successor(diamond, "d2", [
            {"child": "b", "parent": "m", "weight": "1/3"},
            {"child": "b", "parent": "r", "weight": "2/3"},
        ])
        self.assertEqual(new.changed_children, ("b",))
        self.assertEqual(new.recomputed_nodes, ("b", "s"))
        self.assertEqual(new.reused_nodes, ("a", "m", "r"))
        self.assertEqual(new.rows["m"]["r"].computed_in, "d1")
        self.assertEqual(new.bridge_weight("s", "m"), Fraction(1, 2))
        self.assertEqual(new.bridge_weight("s", "r"), 1)
        self.assertEqual(new.verify_bridge()["ok"], True)

    def test_rebind_deltas_and_plan_binding(self):
        result = plan_rebind(self.base, self.new, self.facts, ["f1"])
        self.assertIsInstance(result, RebindResult)
        plan = result.plan
        self.assertEqual(plan.old_version, "hv1")
        self.assertEqual(plan.new_version, "hv2")
        self.assertEqual(plan.old_hierarchy_digest, self.base.digest)
        self.assertEqual(plan.new_hierarchy_digest, self.new.digest)
        self.assertEqual(plan.old_fact_set_digest, self.facts.digest())
        self.assertEqual(plan.rebind_fact_ids, ("f1",))
        self.assertEqual(plan.unrebound_fact_ids, ("f2", "f3"))
        self.assertEqual(
            [fact.hierarchy_version for fact in result.rebound_facts],
            ["hv2", "hv1", "hv1"],
        )
        delta = result.deltas[0].as_dict()
        by_ancestor = {row["ancestor"]: row for row in delta["ancestors"]}
        self.assertEqual(by_ancestor["rootA"]["old_weight"], "3/4")
        self.assertEqual(by_ancestor["rootA"]["new_weight"], "5/8")
        self.assertEqual(by_ancestor["rootA"]["old_contribution"], "75")
        self.assertEqual(by_ancestor["rootA"]["new_contribution"], "125/2")
        self.assertEqual(by_ancestor["rootA"]["delta"], "-25/2")
        self.assertEqual(by_ancestor["rootB"]["delta"], "25/2")
        self.assertEqual(by_ancestor["left"]["delta"], "0")
        self.assertEqual(plan.digest, copy.deepcopy(plan).digest)

    def test_rebind_null_fact_has_no_numeric_delta(self):
        result = plan_rebind(self.base, self.new, self.facts, ["f3"])
        delta = result.deltas[0].as_dict()
        self.assertTrue(delta["missing"])
        self.assertIsNone(delta["value"])
        for row in delta["ancestors"]:
            self.assertIsNone(row["old_contribution"])
            self.assertIsNone(row["new_contribution"])
            self.assertIsNone(row["delta"])
        self.assertEqual(
            [fact.hierarchy_version for fact in result.rebound_facts],
            ["hv1", "hv1", "hv2"],
        )

    def test_rebind_rejects_bad_ids(self):
        with self.assertRaises(ValueError):
            plan_rebind(self.base, self.new, self.facts, [])
        with self.assertRaises(ValueError):
            plan_rebind(self.base, self.new, self.facts, ["ghost"])
        with self.assertRaises(ValueError):
            plan_rebind(self.base, self.new, self.facts, ["f1", "f1"])
        other = Hierarchy.from_manifest(diamond_manifest())
        with self.assertRaises(ValueError):
            plan_rebind(self.base, other, self.facts, ["f1"])  # parent_version 不符
        facts_new = self.facts.with_rebound(["f1"], "hv2")
        with self.assertRaises(ValueError):
            plan_rebind(self.base, self.new, facts_new, ["f1"])  # 已非旧版本

    def test_multi_version_allocation_keeps_versions_separate(self):
        result = plan_rebind(self.base, self.new, self.facts, ["f1"])
        payload = allocate_facts(
            result.rebound_facts, {"hv1": self.base, "hv2": self.new}, include_paths=False
        )
        by_id = {row["fact_id"]: row for row in payload["facts"]}
        self.assertEqual(by_id["f1"]["hierarchy_version"], "hv2")
        self.assertEqual(by_id["f1"]["measured_root_total"], "100")
        self.assertEqual(
            {row["ancestor"]: row["contribution"] for row in by_id["f1"]["contributions"]}["rootA"],
            "125/2",
        )
        self.assertEqual(by_id["f2"]["hierarchy_version"], "hv1")
        self.assertEqual(
            {row["ancestor"]: row["contribution"] for row in by_id["f2"]["contributions"]}["rootA"],
            "-15",
        )
        self.assertEqual(by_id["f3"]["hierarchy_version"], "hv1")


class CommitTests(unittest.TestCase):
    def setUp(self):
        self.manifest, _ = load("data/normal.json")
        self.base = Hierarchy.from_manifest(self.manifest)
        self.facts = FactSet.from_manifest(self.manifest, hierarchy=self.base)
        self.new = Hierarchy.successor(self.base, "hv2", self.manifest["new_edge_weights"])
        self.rebind = plan_rebind(self.base, self.new, self.facts, ["f1"])

    def test_document_round_trip_and_atomic_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "warehouse.json"
            summary = commit_warehouse(
                path, "wh1", self.new, self.facts,
                base_hierarchy=self.base, rebind=self.rebind,
            )
            self.assertEqual(summary["hierarchy_versions"], ["hv1", "hv2"])
            self.assertEqual(summary["rebind"]["rebound_fact_count"], 1)
            self.assertEqual(list(Path(directory).iterdir()), [path])  # 无临时残留
            document, reloaded = load_warehouse(path)
            self.assertEqual(reloaded["digest"], summary["digest"])
            self.assertEqual(document["kind"], "warehouse-version")
            self.assertEqual(
                [fact["hierarchy_version"] for fact in document["facts"]],
                ["hv2", "hv1", "hv1"],
            )
            raw = path.read_bytes()
            self.assertEqual(json.loads(raw.decode("utf8"))["digest"], summary["digest"])

    def test_failure_injection_keeps_prior_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "warehouse.json"
            commit_warehouse(path, "wh0", self.base, self.facts)
            before = path.read_bytes()
            with self.assertRaises(RuntimeError):
                commit_warehouse(
                    path, "wh1", self.new, self.facts,
                    base_hierarchy=self.base, rebind=self.rebind,
                    fault="after_staging",
                )
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            document, summary = load_warehouse(path)
            self.assertEqual(summary["hierarchy_versions"], ["hv1"])
            self.assertIsNone(document["rebind"])

    def test_document_rejects_mixed_new_bridge_and_old_facts(self):
        document = build_warehouse_document(
            "wh1", self.new, self.facts, base_hierarchy=self.base, rebind=self.rebind
        )
        bad = copy.deepcopy(document)
        # f1 仍登记为 hv1 却配 hv2 桥：重挂计划要求它已切到 hv2。
        bad["facts"][0]["hierarchy_version"] = "hv1"
        bad.pop("digest"); bad["digest"] = _document_digest(bad)
        with self.assertRaises(ValueError) as caught:
            verify_warehouse(bad)
        self.assertIn("新桥旧引用", str(caught.exception))

    def test_document_rejects_bridge_from_wrong_version(self):
        document = build_warehouse_document(
            "wh1", self.new, self.facts, base_hierarchy=self.base, rebind=self.rebind
        )
        bad = copy.deepcopy(document)
        hv2_record = bad["hierarchies"][1]
        for source, ancestor, weight in (("leaf", "rootA", "3/4"), ("leaf", "rootB", "1/4"),
                                         ("right", "rootA", "1/2"), ("right", "rootB", "1/2")):
            hv2_record["bridge"][source][ancestor]["weight"] = weight
        bad.pop("digest"); bad["digest"] = _document_digest(bad)
        with self.assertRaises(ValueError) as caught:
            verify_warehouse(bad)
        self.assertIn("桥", str(caught.exception))

    def test_pure_structure_commit_and_prior_version_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "warehouse.json"
            summary = commit_warehouse(path, "wh0", self.base, self.facts)
            self.assertEqual(summary["hierarchy_versions"], ["hv1"])
            document, _ = load_warehouse(path)
            self.assertIsNone(document["rebind"])
            # 已有仓版本不允许被静默覆盖成另一个版本：先读旧摘要再决定目标。
            self.assertNotEqual(summary["digest"], "")
            # 无重挂的文档不得携带多个层级。
            bad = copy.deepcopy(document)
            bad["hierarchies"].append(copy.deepcopy(bad["hierarchies"][0]))
            bad.pop("digest"); bad["digest"] = _document_digest(bad)
            with self.assertRaises(ValueError):
                verify_warehouse(bad)

    def test_document_rejects_tampered_delta_and_value(self):
        document = build_warehouse_document(
            "wh1", self.new, self.facts, base_hierarchy=self.base, rebind=self.rebind
        )
        bad = copy.deepcopy(document)
        bad["rebind"]["deltas"][0]["ancestors"][3]["delta"] = "-20"
        bad.pop("digest"); bad["digest"] = _document_digest(bad)
        with self.assertRaises(ValueError) as caught:
            verify_warehouse(bad)
        self.assertIn("差额", str(caught.exception))

        bad2 = copy.deepcopy(document)
        bad2["facts"][2]["value"] = 0
        bad2.pop("digest"); bad2["digest"] = _document_digest(bad2)
        with self.assertRaises(ValueError):
            verify_warehouse(bad2)

        bad3 = copy.deepcopy(document)
        bad3.pop("digest")
        with self.assertRaises(ValueError):
            verify_warehouse(bad3)

    def test_unrebound_facts_keep_old_version_in_document(self):
        document = build_warehouse_document(
            "wh1", self.new, self.facts, base_hierarchy=self.base, rebind=self.rebind
        )
        plan = document["rebind"]["plan"]
        self.assertEqual(plan["rebind_fact_ids"], ["f1"])
        self.assertEqual(plan["unrebound_fact_ids"], ["f2", "f3"])
        self.assertEqual(plan["new_hierarchy_summary"]["digest"], self.new.digest)
        self.assertEqual(
            [fact["hierarchy_version"] for fact in document["facts"]], ["hv2", "hv1", "hv1"]
        )

    def test_missing_new_hierarchy_rejected(self):
        with self.assertRaises(ValueError):
            build_warehouse_document("wh1", self.base, self.facts,
                                     base_hierarchy=self.base, rebind=self.rebind)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        out = StringIO(); err = StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_inspect_and_rollup(self):
        code, out, err = self.run_cli("inspect", "data/normal.json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["bridge_check"]["ok"])
        self.assertEqual(payload["hierarchy"]["version"], "hv1")

        code, out, _ = self.run_cli("rollup", "data/normal.json")
        payload = json.loads(out)
        self.assertTrue(payload["safe_to_add"])
        self.assertEqual(payload["group_total"], "80")

        code, out, _ = self.run_cli("rollup", "data/nonadditive-selection.json")
        payload = json.loads(out)
        self.assertFalse(payload["safe_to_add"])
        self.assertIsNone(payload["group_total"])

    def test_invalid_data_exits_nonzero(self):
        code, _, err = self.run_cli("inspect", "data/invalid-config.json")
        self.assertEqual(code, 2)
        self.assertIn("error", json.loads(err))

    def test_commit_and_verify_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wh.json"
            code, _, _ = self.run_cli(
                "commit", "data/normal.json", "--out", str(path),
                "--warehouse-version", "wh1", "--new-version", "hv2",
            )
            self.assertEqual(code, 0)
            code, out, _ = self.run_cli("verify", str(path))
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["hierarchy_versions"], ["hv1", "hv2"])


class NoSilentTreeFixTests(unittest.TestCase):
    def test_multi_parent_is_never_reduced_to_single_parent(self):
        hierarchy = Hierarchy.from_manifest(skip_manifest())
        # s 有两个父（a 和 r），r 也是根；零权直接边不得从结构或证据中消失。
        self.assertEqual(hierarchy.parent_counts()["s"], 2)
        self.assertEqual(len(hierarchy.parent_edges("s")), 2)
        entry = hierarchy.bridge_entry("s", "r")
        self.assertIn(("s", "r"), {path.nodes for path in entry.paths})
        # 归一化分量：leaf 经两条父路得到 1/2 与 1/2，而非全部分给单一父。
        normal, _ = load("data/normal.json")
        h = Hierarchy.from_manifest(normal)
        direct = {edge.parent: edge.weight for edge in h.parent_edges("leaf")}
        self.assertEqual(direct, {"left": Fraction(1, 2), "right": Fraction(1, 2)})
        self.assertEqual(h.bridge_weight("leaf", "rootA"), Fraction(3, 4))
        self.assertNotEqual(h.bridge_weight("leaf", "rootA"), 1)

    def test_roots_may_exceed_one_node_and_include_isolated(self):
        normal, _ = load("data/normal.json")
        h = Hierarchy.from_manifest(normal)
        self.assertEqual(h.roots, ("rootA", "rootB", "unrelated"))
        self.assertEqual(len(h.roots), 3)
        self.assertEqual(h.parent_counts()["unrelated"], 0)
        for node in h.nodes:
            self.assertEqual(
                sum((entry.weight for entry in h.rows[node].values()
                     if entry.ancestor in h.roots), Fraction(0)),
                1,
            )


class ClosureHelperTests(unittest.TestCase):
    def test_closure_matches_successor_recomputed_set(self):
        for manifest, updates in (
            (diamond_manifest(), [
                {"child": "b", "parent": "m", "weight": "1/3"},
                {"child": "b", "parent": "r", "weight": "2/3"},
            ]),
            (skip_manifest(), [
                {"child": "s", "parent": "a", "weight": "1/2"},
                {"child": "s", "parent": "r", "weight": "1/2"},
            ]),
        ):
            base = Hierarchy.from_manifest(manifest)
            new = Hierarchy.successor(base, base.version + "x", updates)
            self.assertEqual(
                base.closure_of_changed_children(new.changed_children),
                new.recomputed_nodes,
            )
        with self.assertRaises(ValueError):
            Hierarchy.from_manifest(diamond_manifest()).closure_of_changed_children(["ghost"])


class MaterialRegressionTests(unittest.TestCase):
    def test_all_valid_materials_build_bridges(self):
        for path in Path("data").glob("*.json"):
            if path.stem == "invalid-config":
                continue
            with self.subTest(file=path.name):
                manifest = load(path)[0]
                if path.stem == "structural-cycle":
                    with self.assertRaises(ValueError) as caught:
                        Hierarchy.from_manifest(manifest)
                    self.assertIn("环", str(caught.exception))
                    continue
                hierarchy = Hierarchy.from_manifest(manifest)
                self.assertTrue(hierarchy.verify_bridge()["ok"])
                facts = FactSet.from_manifest(manifest, hierarchy=hierarchy)
                self.assertEqual(len(facts), 3)
                try:
                    rollup = assess_rollup(facts, hierarchy, manifest["selected_rollup_nodes"])
                except ValueError as exc:
                    self.assertEqual(path.stem, "structural-cycle", msg=str(exc))
                    continue
                self.assertEqual(
                    rollup["safe_to_add"], path.stem == "normal",
                    msg=rollup["reasons"],
                )
                if path.stem == "normal":
                    self.assertEqual(rollup["group_total"], "80")


if __name__ == "__main__":
    unittest.main()
