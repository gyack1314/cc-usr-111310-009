"""事实、按绑定版本的祖先分摊、非重叠汇总与原子重挂提交。

核心约束：

* 每个事实带稳定 ``fact_id``、原始叶节点、定标整数值和
  ``hierarchy_version``；分摊严格按该版本桥计算。
* ``value: null`` 表示事实存在但数值缺测，不按 ``0`` 参与守恒；
  负值是冲正，照常保留。
* 多个根各自得到同一事实的一部分，跨全部根求和才等于事实值；
  只有用户明确选择且正好覆盖全部根时才输出集团总量。
* 重挂计划绑定旧事实集合摘要与新层级摘要；层级、桥、重挂及差额在
  一次仓版本提交中原子替换，失败不留下新旧混用状态。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .hierarchy import (
    BridgePath,
    Hierarchy,
    format_weight,
    parse_plain_integer,
)

FACT_KIND = "warehouse-version"


# ---------------------------------------------------------------------------
# 身份、序列化与摘要
# ---------------------------------------------------------------------------


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


_SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


def _require_text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what}必须是非空字符串")
    return value


def _require_slug(value: object, what: str) -> str:
    """仓版本号会出现在文件名候选与外部引用中，限制为安全标识符。"""
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value):
        raise ValueError(f"{what}必须是不含路径分隔符的标识符")
    return value


@dataclass(frozen=True, slots=True)
class Fact:
    fact_id: str
    node: str
    value: int | None
    hierarchy_version: str

    def as_dict(self) -> dict:
        # 缺测保留显式 null，不省略字段。
        return {
            "fact_id": self.fact_id,
            "node": self.node,
            "value": self.value,
            "hierarchy_version": self.hierarchy_version,
        }

    def rebound(self, version: str) -> "Fact":
        return replace(self, hierarchy_version=version)


def parse_fact(row: object, *, nodes: Iterable[str] | None = None,
               known_versions: Iterable[str] | None = None) -> Fact:
    if not isinstance(row, Mapping):
        raise ValueError("事实必须是对象")
    fact_id = _require_text(row.get("fact_id"), "fact_id")
    node = _require_text(row.get("node"), "事实节点")
    version = _require_text(row.get("hierarchy_version"), "事实层级版本")
    if nodes is not None and node not in set(nodes):
        raise ValueError(f"事实 {fact_id} 引用未知节点 {node}")
    if known_versions is not None and version not in set(known_versions):
        raise ValueError(f"事实 {fact_id} 引用未知层级版本 {version}")
    value = row.get("value")
    if value is not None:
        value = parse_plain_integer(value, f"事实 {fact_id} 的值")
    return Fact(fact_id, node, value, version)


@dataclass(frozen=True, slots=True)
class FactSet:
    facts: tuple[Fact, ...]

    @classmethod
    def from_rows(cls, rows: Sequence[object] | Iterable[object], *,
                  nodes: Iterable[str] | None = None,
                  known_versions: Iterable[str] | None = None) -> "FactSet":
        if not isinstance(rows, (list, tuple)):
            raise ValueError("事实清单必须是数组")
        facts: list[Fact] = []
        seen: set[str] = set()
        for row in rows:
            fact = parse_fact(row, nodes=nodes, known_versions=known_versions)
            if fact.fact_id in seen:
                raise ValueError(f"fact_id 重复: {fact.fact_id}")
            seen.add(fact.fact_id)
            facts.append(fact)
        facts.sort(key=lambda f: f.fact_id)
        return cls(tuple(facts))

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object], *,
                      hierarchy: Hierarchy | None = None,
                      known_versions: Iterable[str] | None = None) -> "FactSet":
        if not isinstance(manifest, Mapping):
            raise ValueError("清单必须是对象")
        if hierarchy is not None:
            known_versions = {hierarchy.version}
            return cls.from_rows(
                manifest.get("facts"), nodes=hierarchy.nodes,
                known_versions=known_versions,
            )
        return cls.from_rows(manifest.get("facts"), known_versions=known_versions)

    def __iter__(self) -> Iterable[Fact]:
        return iter(self.facts)

    def __len__(self) -> int:
        return len(self.facts)

    @property
    def by_id(self) -> dict[str, Fact]:
        return {fact.fact_id: fact for fact in self.facts}

    def require(self, fact_id: str) -> Fact:
        try:
            return self.by_id[fact_id]
        except KeyError as exc:
            raise ValueError(f"未知 fact_id: {fact_id}") from exc

    def as_dicts(self) -> list[dict]:
        return [fact.as_dict() for fact in self.facts]

    def digest(self) -> str:
        return canonical_digest(self.as_dicts())

    def with_rebound(self, fact_ids: Sequence[str], version: str) -> "FactSet":
        ids = set(fact_ids)
        return FactSet(tuple(
            fact.rebound(version) if fact.fact_id in ids else fact
            for fact in self.facts
        ))

    def missing(self) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.value is None)

    def measured(self) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.value is not None)


# ---------------------------------------------------------------------------
# 事实按桥分摊
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Contribution:
    ancestor: str
    weight: Fraction
    value: int | None  # None 表示 fact 缺测；不是 0
    contribution: Fraction | None
    paths: tuple[BridgePath, ...]

    def as_dict(self) -> dict:
        return {
            "ancestor": self.ancestor,
            "weight": format_weight(self.weight),
            "value": self.value,
            "contribution": None if self.contribution is None else format_weight(self.contribution),
            "paths": [path.as_dict() for path in self.paths],
        }


@dataclass(frozen=True, slots=True)
class FactAllocation:
    fact_id: str
    node: str
    value: int | None
    hierarchy_version: str
    contributions: tuple[Contribution, ...]
    missing: bool
    measured_root_total: Fraction | None
    root_total_equals_value: bool | None

    def as_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "node": self.node,
            "value": self.value,
            "hierarchy_version": self.hierarchy_version,
            "missing": self.missing,
            "measured_root_total": None if self.measured_root_total is None
            else format_weight(self.measured_root_total),
            "root_total_equals_value": self.root_total_equals_value,
            "contributions": [row.as_dict() for row in self.contributions],
        }

    def contribution(self, ancestor: str) -> Fraction | None:
        for row in self.contributions:
            if row.ancestor == ancestor:
                return row.contribution
        return Fraction(0) if self.value is not None else None


def allocate_fact(fact: Fact, hierarchy: Hierarchy, *,
                  include_paths: bool = True) -> FactAllocation:
    if fact.hierarchy_version != hierarchy.version:
        raise ValueError(
            f"事实 {fact.fact_id} 绑定 {fact.hierarchy_version}，"
            f"不能用 {hierarchy.version} 的桥分摊"
        )
    if fact.node not in set(hierarchy.nodes):
        raise ValueError(f"事实 {fact.fact_id} 引用当前层级未知节点 {fact.node}")
    rows = hierarchy.rows[fact.node]
    contributions = []
    for ancestor, entry in rows.items():
        exact = None if fact.value is None else Fraction(fact.value) * entry.weight
        contributions.append(Contribution(
            ancestor=ancestor,
            weight=entry.weight,
            value=fact.value,
            contribution=exact,
            paths=entry.paths if include_paths else (),
        ))
    # 祖先按 nodes 声明顺序稳定排序，便于阅读连续输出；该顺序不代表
    # 拓扑深度（层级不是固定深度），也不影响求和。
    order = {name: index for index, name in enumerate(hierarchy.nodes)}
    contributions.sort(key=lambda row: (order.get(row.ancestor, len(order)), row.ancestor))
    root_total = None
    equals_value = None
    if fact.value is not None:
        root_total = sum(
            (row.contribution for row in contributions
             if row.ancestor in hierarchy.roots and row.contribution is not None),
            Fraction(0),
        )
        equals_value = root_total == Fraction(fact.value)
    return FactAllocation(
        fact_id=fact.fact_id,
        node=fact.node,
        value=fact.value,
        hierarchy_version=fact.hierarchy_version,
        contributions=tuple(contributions),
        missing=fact.value is None,
        measured_root_total=root_total,
        root_total_equals_value=equals_value,
    )


def allocate_facts(facts: FactSet, hierarchies: Hierarchy | Mapping[str, Hierarchy],
                   *, include_paths: bool = True) -> dict:
    mapping = _hierarchy_map(hierarchies)
    allocations = []
    for fact in facts:
        hierarchy = _hierarchy_for(mapping, fact.hierarchy_version, fact)
        allocations.append(allocate_fact(fact, hierarchy, include_paths=include_paths))
    measured = sum(1 for row in allocations if not row.missing)
    missing = len(allocations) - measured
    conserved = sum(1 for row in allocations if row.root_total_equals_value is True)
    violated = [
        row.fact_id for row in allocations if row.root_total_equals_value is False
    ]
    if violated:
        raise ValueError(f"事实根贡献不守恒: {violated}")
    return {
        "facts": [row.as_dict() for row in allocations],
        "measured_fact_count": measured,
        "missing_fact_count": missing,
        "conserved_measured_fact_count": conserved,
    }


def _hierarchy_map(hierarchies: Hierarchy | Mapping[str, Hierarchy]) -> dict[str, Hierarchy]:
    if isinstance(hierarchies, Hierarchy):
        return {hierarchies.version: hierarchies}
    if not isinstance(hierarchies, Mapping):
        raise ValueError("层级集合必须是映射或单个层级")
    result: dict[str, Hierarchy] = {}
    for version, hierarchy in hierarchies.items():
        if not isinstance(hierarchy, Hierarchy):
            raise ValueError("层级集合包含非层级对象")
        if version != hierarchy.version:
            raise ValueError("层级集合的键必须等于层级版本")
        result[version] = hierarchy
    if not result:
        raise ValueError("层级集合不能为空")
    return result


def _hierarchy_for(mapping: Mapping[str, Hierarchy], version: str, fact: Fact) -> Hierarchy:
    try:
        return mapping[version]
    except KeyError as exc:
        raise ValueError(
            f"事实 {fact.fact_id} 绑定的层级版本 {version} 不在本次解析范围内"
        ) from exc


# ---------------------------------------------------------------------------
# 显式选择节点的汇总与非重叠判断
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeRollup:
    node: str
    total: Fraction
    measured_fact_count: int
    missing_fact_count: int

    def as_dict(self) -> dict:
        return {
            "node": self.node,
            "total": format_weight(self.total),
            "measured_fact_count": self.measured_fact_count,
            "missing_fact_count": self.missing_fact_count,
        }


def assess_rollup(facts: FactSet, hierarchies: Hierarchy | Mapping[str, Hierarchy],
                  selected_nodes: Sequence[str]) -> dict:
    """按每个事实绑定的层级版本计算所选节点贡献，并判断能否直接相加。

    ``group_total`` 只在所选节点正好是全部根时输出。其它选择即使每个
    单节点数字都能算，也只作为候选证据返回，不冒充集团总量。
    """
    mapping = _hierarchy_map(hierarchies)
    if not isinstance(selected_nodes, (list, tuple)) or not selected_nodes:
        raise ValueError("所选汇总节点必须是非空数组")
    selected: list[str] = []
    for node in selected_nodes:
        _require_text(node, "汇总节点")
        if node in selected:
            raise ValueError(f"汇总节点重复: {node}")
        selected.append(node)
    selected_set = set(selected)

    used_versions = {fact.hierarchy_version for fact in facts}
    if not used_versions:
        raise ValueError("没有事实可供汇总")
    for version in used_versions:
        hierarchy = mapping.get(version)
        if hierarchy is None:
            raise ValueError(f"缺少事实绑定的层级版本 {version}")
        unknown = [node for node in selected if node not in hierarchy.nodes]
        if unknown:
            raise ValueError(f"汇总节点 {unknown} 在层级版本 {version} 中不存在")

    reasons: list[str] = []
    ancestor_pairs: list[dict] = []
    # 祖先关系按所有涉及层级逐个检查；任一层级成立即视为重叠。
    for index, left in enumerate(selected):
        for right in selected[index + 1:]:
            for hierarchy in mapping.values():
                if left not in hierarchy.nodes or right not in hierarchy.nodes:
                    continue
                if hierarchy.is_ancestor(left, right):
                    ancestor_pairs.append(
                        {"ancestor": left, "descendant": right, "version": hierarchy.version}
                    )
                    break
                if hierarchy.is_ancestor(right, left):
                    ancestor_pairs.append(
                        {"ancestor": right, "descendant": left, "version": hierarchy.version}
                    )
                    break
    # 每个事实按自身版本的桥分摊给所有选中节点，记录真正有多节点贡献的事实。
    overlap_facts: list[dict] = []
    uncovered_facts: list[dict] = []
    per_node = {node: {"total": Fraction(0), "measured": 0, "missing": 0} for node in selected}
    fact_rows: list[dict] = []
    conservation_ok = True
    for fact in facts:
        hierarchy = mapping[fact.hierarchy_version]
        entries = hierarchy.rows[fact.node]
        roots = hierarchy.roots
        root_weights = [entries[root].weight for root in roots if root in entries]
        if fact.value is not None and sum(root_weights, Fraction(0)) != 1:
            conservation_ok = False
        fact_selected = []
        selected_root_coverage = Fraction(0)
        for node in selected:
            weight = entries[node].weight if node in entries else Fraction(0)
            contribution = None if fact.value is None else Fraction(fact.value) * weight
            if weight > 0:
                fact_selected.append(node)
            if node in roots:
                selected_root_coverage += weight
            bucket = per_node[node]
            if node in entries:
                if fact.value is None:
                    bucket["missing"] += 1
                else:
                    bucket["total"] += contribution
                    bucket["measured"] += 1
        # 覆盖要求：该事实在各选中根上的桥权重之和必须为1。缺测事实的
        # 数值不参与守恒，但其结构覆盖仍按桥权重检查。
        if selected_root_coverage != 1:
            uncovered_facts.append({
                "fact_id": fact.fact_id,
                "node": fact.node,
                "hierarchy_version": fact.hierarchy_version,
                "selected_root_coverage": format_weight(selected_root_coverage),
            })
        if len(fact_selected) > 1:
            overlap_facts.append({
                "fact_id": fact.fact_id,
                "node": fact.node,
                "contributing_selected_nodes": fact_selected,
            })
        fact_rows.append({
            "fact_id": fact.fact_id,
            "node": fact.node,
            "value": fact.value,
            "hierarchy_version": fact.hierarchy_version,
            "contributing_selected_nodes": fact_selected,
        })

    # 直接可加的条件（三者同时成立）：
    # 1. 每个选中节点在每个相关层级里都是根（根没有父边，避免把已在
    #    祖先根份额中的中间层贡献再加一次）；
    # 2. 所选节点内部没有祖先/后代关系；
    # 3. 每个事实流向选中根的桥权重和恰好为1（覆盖全部根贡献，不漏不重）。
    # 无事实的孤立根不必被强制选中，因为它对任何事实的贡献都是零。
    root_membership = all(
        all(node in set(mapping[version].roots) for version in used_versions)
        for node in selected
    )
    coverage_ok = not uncovered_facts and conservation_ok
    safe_to_add = bool(root_membership and coverage_ok and not ancestor_pairs)
    if not safe_to_add:
        if not root_membership:
            reasons.append(
                "所选节点中存在非根节点；中间层贡献已包含在其祖先根的份额里，不能与根一起相加。"
            )
        if ancestor_pairs:
            reasons.append(
                "所选节点集合内部存在祖先/后代关系，单节点贡献之和会重复计入同一事实。"
            )
        if overlap_facts:
            reasons.append(
                "所选节点集合存在共同下游事实，单节点贡献之和会重复计入这些事实。"
            )
        if uncovered_facts:
            reasons.append(
                "所选根集合没有覆盖某些事实的全部根贡献，其桥权重之和不为1。"
            )
    coverage_proven = safe_to_add

    candidate_sum = sum((bucket["total"] for bucket in per_node.values()), Fraction(0))
    measured_total = sum(Fraction(fact.value) for fact in facts if fact.value is not None)
    result = {
        "selected_nodes": selected,
        "used_hierarchy_versions": sorted(used_versions),
        "safe_to_add": safe_to_add,
        "coverage_proven": coverage_proven,
        "conservation_ok": conservation_ok,
        "can_be_group_total": safe_to_add and coverage_proven,
        "group_total": None,
        "reasons": reasons,
        "ancestor_pairs": ancestor_pairs,
        "overlap_facts": overlap_facts,
        "uncovered_facts": uncovered_facts,
        "root_membership": root_membership,
        "root_coverage_ok": coverage_ok,
        "per_node": {node: {
            "total": format_weight(bucket["total"]),
            "measured_fact_count": bucket["measured"],
            "missing_fact_count": bucket["missing"],
        } for node, bucket in per_node.items()},
        "candidate_unqualified_sum": format_weight(candidate_sum),
        "fact_value_total": format_weight(measured_total),
        "measured_fact_count": len(facts.measured()),
        "missing_fact_count": len(facts.missing()),
        "fact_rows": fact_rows,
    }
    if safe_to_add and coverage_proven:
        # 根集合覆盖全部事实后，各根贡献和精确等于所有可加事实值之和；
        # 若不等则说明内部不变量已被破坏，宁可拒绝也不输出错误总量。
        if candidate_sum != measured_total:
            raise ValueError("集团总量与事实值总和不一致，拒绝输出")
        result["group_total"] = format_weight(candidate_sum)
        result["group_total_equals_fact_value_total"] = True
    return result


# ---------------------------------------------------------------------------
# 重挂计划与差额
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RebindAncestor:
    ancestor: str
    old_weight: Fraction
    new_weight: Fraction
    old_contribution: Fraction | None
    new_contribution: Fraction | None
    delta: Fraction | None

    def as_dict(self) -> dict:
        return {
            "ancestor": self.ancestor,
            "old_weight": format_weight(self.old_weight),
            "new_weight": format_weight(self.new_weight),
            "old_contribution": None if self.old_contribution is None else format_weight(self.old_contribution),
            "new_contribution": None if self.new_contribution is None else format_weight(self.new_contribution),
            "delta": None if self.delta is None else format_weight(self.delta),
        }


@dataclass(frozen=True, slots=True)
class RebindFactDelta:
    fact_id: str
    node: str
    value: int | None
    old_version: str
    new_version: str
    missing: bool
    ancestors: tuple[RebindAncestor, ...]

    def as_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "node": self.node,
            "value": self.value,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "missing": self.missing,
            "ancestors": [row.as_dict() for row in self.ancestors],
        }


@dataclass(frozen=True, slots=True)
class RebindPlan:
    old_version: str
    new_version: str
    old_hierarchy_digest: str
    new_hierarchy_digest: str
    old_hierarchy_summary: dict
    new_hierarchy_summary: dict
    old_fact_set_digest: str
    rebind_facts_digest: str
    rebind_fact_ids: tuple[str, ...]
    unrebound_fact_ids: tuple[str, ...]
    old_facts: tuple[Fact, ...]
    digest: str

    def as_dict(self) -> dict:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "old_hierarchy_digest": self.old_hierarchy_digest,
            "new_hierarchy_digest": self.new_hierarchy_digest,
            "old_hierarchy_summary": self.old_hierarchy_summary,
            "new_hierarchy_summary": self.new_hierarchy_summary,
            "old_fact_set_digest": self.old_fact_set_digest,
            "rebind_facts_digest": self.rebind_facts_digest,
            "rebind_fact_ids": list(self.rebind_fact_ids),
            "unrebound_fact_ids": list(self.unrebound_fact_ids),
            "old_facts": [fact.as_dict() for fact in self.old_facts],
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class RebindResult:
    plan: RebindPlan
    deltas: tuple[RebindFactDelta, ...]
    rebound_facts: FactSet

    def deltas_as_dicts(self) -> list[dict]:
        return [row.as_dict() for row in self.deltas]


def plan_rebind(base: Hierarchy, new: Hierarchy, facts: FactSet,
                rebind_fact_ids: Sequence[str]) -> RebindResult:
    """为指定事实生成旧新祖先差额计划；未列出的事实一律保持原版本。"""
    if base.version != new.parent_version:
        raise ValueError(
            f"新层级 {new.version} 的父版本 {new.parent_version} 与旧层级 {base.version} 不符"
        )
    if not isinstance(rebind_fact_ids, (list, tuple)) or not rebind_fact_ids:
        raise ValueError("rebind_fact_ids 必须是非空数组，显式重挂才允许列事实")
    ids: list[str] = []
    for fact_id in rebind_fact_ids:
        _require_text(fact_id, "rebind_fact_id")
        if fact_id in ids:
            raise ValueError(f"重挂 fact_id 重复: {fact_id}")
        ids.append(fact_id)
    for fact_id in ids:
        fact = facts.require(fact_id)
        if fact.hierarchy_version != base.version:
            raise ValueError(
                f"事实 {fact_id} 绑定 {fact.hierarchy_version}，不能按 {base.version} 重挂"
            )

    by_id = facts.by_id
    rebound = facts.with_rebound(ids, new.version)
    deltas: list[RebindFactDelta] = []
    unrebound_ids = sorted(set(by_id) - set(ids))
    for fact_id in ids:
        old_fact = by_id[fact_id]
        new_fact = rebound.require(fact_id)
        old_alloc = allocate_fact(old_fact, base, include_paths=False)
        new_alloc = allocate_fact(new_fact, new, include_paths=False)
        old_contribs = {row.ancestor: row.contribution for row in old_alloc.contributions}
        new_contribs = {row.ancestor: row.contribution for row in new_alloc.contributions}
        old_weights = {row.ancestor: row.weight for row in old_alloc.contributions}
        new_weights = {row.ancestor: row.weight for row in new_alloc.contributions}
        ancestor_names = sorted(set(old_weights) | set(new_weights))
        rows = []
        for ancestor in ancestor_names:
            old_contribution = old_contribs.get(ancestor)
            new_contribution = new_contribs.get(ancestor)
            if old_fact.value is None:
                delta = None
            else:
                delta = (new_contribution or Fraction(0)) - (old_contribution or Fraction(0))
            rows.append(RebindAncestor(
                ancestor=ancestor,
                old_weight=old_weights.get(ancestor, Fraction(0)),
                new_weight=new_weights.get(ancestor, Fraction(0)),
                old_contribution=old_contribution,
                new_contribution=new_contribution,
                delta=delta,
            ))
        deltas.append(RebindFactDelta(
            fact_id=fact_id,
            node=old_fact.node,
            value=old_fact.value,
            old_version=base.version,
            new_version=new.version,
            missing=old_fact.value is None,
            ancestors=tuple(rows),
        ))

    old_rows = tuple(facts.require(fact_id) for fact_id in sorted(ids))
    plan_body = {
        "old_version": base.version,
        "new_version": new.version,
        "old_hierarchy_digest": base.digest,
        "new_hierarchy_digest": new.digest,
        "old_hierarchy_summary": base.summary(),
        "new_hierarchy_summary": new.summary(),
        "old_fact_set_digest": facts.digest(),
        "rebind_facts_digest": canonical_digest([fact.as_dict() for fact in old_rows]),
        "rebind_fact_ids": sorted(ids),
        "unrebound_fact_ids": unrebound_ids,
        "old_facts": [fact.as_dict() for fact in facts.facts],
    }
    plan = RebindPlan(
        old_version=base.version,
        new_version=new.version,
        old_hierarchy_digest=base.digest,
        new_hierarchy_digest=new.digest,
        old_hierarchy_summary=base.summary(),
        new_hierarchy_summary=new.summary(),
        old_fact_set_digest=facts.digest(),
        rebind_facts_digest=plan_body["rebind_facts_digest"],
        rebind_fact_ids=tuple(sorted(ids)),
        unrebound_fact_ids=tuple(unrebound_ids),
        old_facts=tuple(facts.facts),
        digest=canonical_digest(plan_body),
    )
    return RebindResult(plan=plan, deltas=tuple(deltas), rebound_facts=rebound)


# ---------------------------------------------------------------------------
# 仓版本文档、原子提交与重载校验
# ---------------------------------------------------------------------------


def hierarchy_record(hierarchy: Hierarchy) -> dict:
    return {
        "version": hierarchy.version,
        "parent_version": hierarchy.parent_version,
        "digest": hierarchy.digest,
        "summary": hierarchy.summary(),
        "nodes": list(hierarchy.nodes),
        "edges": hierarchy.edge_as_dicts(),
        "roots": list(hierarchy.roots),
        "bridge": hierarchy.bridge_as_dict(),
    }


def _document_digest(document: Mapping[str, object]) -> str:
    body = {key: value for key, value in document.items() if key != "digest"}
    return canonical_digest(body)


def build_warehouse_document(
    warehouse_version: str,
    hierarchy: Hierarchy,
    facts: FactSet,
    *,
    base_hierarchy: Hierarchy | None = None,
    rebind: RebindResult | None = None,
) -> dict:
    _require_slug(warehouse_version, "仓版本")
    if not isinstance(hierarchy, Hierarchy):
        raise ValueError("当前层级必须是 Hierarchy")
    records: list[dict] = []
    if rebind is not None:
        if base_hierarchy is None:
            raise ValueError("重挂提交必须同时提供旧层级")
        if base_hierarchy.version != rebind.plan.old_version:
            raise ValueError("旧层级版本与重挂计划不符")
        if hierarchy.version != rebind.plan.new_version:
            raise ValueError("当前层级版本与重挂计划不符")
        if hierarchy.digest != rebind.plan.new_hierarchy_digest:
            raise ValueError("当前层级摘要与重挂计划不符")
        if base_hierarchy.digest != rebind.plan.old_hierarchy_digest:
            raise ValueError("旧层级摘要与重挂计划不符")
        if facts.digest() != rebind.plan.old_fact_set_digest:
            raise ValueError("提交的旧事实集合与重挂计划不符")
        rebound_ids = {row.fact_id for row in rebind.deltas}
        if rebound_ids != set(rebind.plan.rebind_fact_ids):
            raise ValueError("重挂差额与计划 fact_id 不一致")
        old_facts = rebind.plan.old_facts
        if canonical_digest([fact.as_dict() for fact in old_facts]) != facts.digest():
            raise ValueError("提交的旧事实集合与重挂计划记录不符")
        # 未显式重挂的事实保持原版本；显式重挂的事实才切到新版本。
        final_facts = rebind.rebound_facts
        records = [hierarchy_record(base_hierarchy), hierarchy_record(hierarchy)]
    else:
        final_facts = facts
        records = [hierarchy_record(hierarchy)]
    document = {
        "kind": FACT_KIND,
        "warehouse_version": warehouse_version,
        "hierarchies": records,
        "facts": final_facts.as_dicts(),
        "rebind": None if rebind is None else {
            "plan": rebind.plan.as_dict(),
            "deltas": rebind.deltas_as_dicts(),
            "rebound_fact_count": len(rebind.deltas),
            "unrebound_fact_count": len(rebind.plan.unrebound_fact_ids),
        },
    }
    # 先算摘要再走与读取端完全相同的重算核对；构建阶段不使用任何
    # “已知可信”捷径，保证写出的文档从第一天就能被独立验证。
    document["digest"] = _document_digest(document)
    verify_warehouse(document)
    return document


def verify_warehouse(document: Mapping[str, object]) -> dict:
    """独立重算层级、桥、事实分摊与重挂差额；任何新旧混用都拒绝。"""
    if not isinstance(document, Mapping):
        raise ValueError("仓版本必须是对象")
    if document.get("kind") != FACT_KIND:
        raise ValueError("不支持的仓版本类型")
    if not isinstance(document.get("digest"), str):
        raise ValueError("仓版本缺少摘要")
    if _document_digest(document) != document["digest"]:
        raise ValueError("仓版本摘要不符")
    warehouse_version = _require_text(document.get("warehouse_version"), "仓版本")

    raw_hierarchies = document.get("hierarchies")
    if not isinstance(raw_hierarchies, (list, tuple)) or not raw_hierarchies:
        raise ValueError("仓版本必须包含层级")
    hierarchies: dict[str, Hierarchy] = {}
    for record in raw_hierarchies:
        if not isinstance(record, Mapping):
            raise ValueError("层级记录必须是对象")
        version = _require_text(record.get("version"), "层级版本")
        if version in hierarchies:
            raise ValueError(f"层级版本重复: {version}")
        hierarchy = Hierarchy.from_mapping(record)
        if hierarchy.version != version:
            raise ValueError("层级版本字段不一致")
        if record.get("digest") != hierarchy.digest:
            raise ValueError(f"层级 {version} 摘要不符")
        if record.get("roots") != list(hierarchy.roots):
            raise ValueError(f"层级 {version} 根集合不符")
        if record.get("edges") != hierarchy.edge_as_dicts():
            raise ValueError(f"层级 {version} 边不符")
        if record.get("bridge") != hierarchy.bridge_as_dict():
            raise ValueError(f"层级 {version} 桥不符（可能新旧桥混用）")
        summary = record.get("summary")
        if not isinstance(summary, Mapping) or summary.get("digest") != hierarchy.digest:
            raise ValueError(f"层级 {version} 摘要记录不符")
        parent_version = record.get("parent_version")
        if parent_version is not None:
            if not isinstance(parent_version, str) or parent_version not in hierarchies:
                raise ValueError(f"层级 {version} 的父层级 {parent_version} 未提交或顺序错误")
            if hierarchy.parent_version != parent_version:
                raise ValueError("层级父子关系字段不一致")
        hierarchies[version] = hierarchy

    raw_facts = document.get("facts")
    facts = FactSet.from_rows(raw_facts, known_versions=hierarchies.keys())
    for fact in facts:
        hierarchy = hierarchies[fact.hierarchy_version]
        if fact.node not in set(hierarchy.nodes):
            raise ValueError(f"事实 {fact.fact_id} 的节点 {fact.node} 不在绑定层级 {fact.hierarchy_version}")

    # 事实节点必须真实存在于其绑定版本；仅按节点名比较无法发现跨版本
    # 引用（两个版本节点集合可能同名但边不同）。版本本身也须已提交。
    for fact in facts:
        if fact.hierarchy_version not in hierarchies:
            raise ValueError(f"事实 {fact.fact_id} 绑定的层级版本未提交")

    raw_rebind = document.get("rebind")
    if raw_rebind is None:
        if len(hierarchies) != 1:
            raise ValueError("无重挂的仓版本不应包含多个层级")
        hierarchy = next(iter(hierarchies.values()))
        if any(fact.hierarchy_version != hierarchy.version for fact in facts):
            raise ValueError("事实版本与仓版本层级不符")
        return {
            "warehouse_version": warehouse_version,
            "digest": document.get("digest"),
            "hierarchy_versions": sorted(hierarchies),
            "facts": len(facts),
            "rebind": None,
        }

    if not isinstance(raw_rebind, Mapping):
        raise ValueError("rebind 必须是对象或 null")
    plan_raw = raw_rebind.get("plan")
    deltas_raw = raw_rebind.get("deltas")
    if not isinstance(plan_raw, Mapping) or not isinstance(deltas_raw, (list, tuple)):
        raise ValueError("rebind 必须包含 plan 和 deltas")
    old_version = _require_text(plan_raw.get("old_version"), "旧层级版本")
    new_version = _require_text(plan_raw.get("new_version"), "新层级版本")
    if old_version not in hierarchies or new_version not in hierarchies:
        raise ValueError("重挂计划引用的层级未提交")
    base = hierarchies[old_version]
    new = hierarchies[new_version]
    if plan_raw.get("old_hierarchy_digest") != base.digest:
        raise ValueError("重挂计划的旧层级摘要不符")
    if plan_raw.get("new_hierarchy_digest") != new.digest:
        raise ValueError("重挂计划的新层级摘要不符")
    summary = plan_raw.get("new_hierarchy_summary")
    if not isinstance(summary, Mapping) or summary.get("digest") != new.digest:
        raise ValueError("重挂计划未绑定新层级摘要")

    old_rows = plan_raw.get("old_facts")
    old_facts = FactSet.from_rows(old_rows, known_versions={old_version})
    if old_facts.digest() != plan_raw.get("old_fact_set_digest"):
        raise ValueError("重挂计划绑定的旧事实集合摘要不符")
    for fact in old_facts:
        if fact.hierarchy_version != old_version:
            raise ValueError("重挂计划中的旧事实版本不符")
    rebind_ids = plan_raw.get("rebind_fact_ids")
    if not isinstance(rebind_ids, (list, tuple)):
        raise ValueError("重挂计划缺少 rebind_fact_ids")
    rebound_ids = {item for item in rebind_ids if isinstance(item, str)}
    if rebound_ids != set(rebind_ids) or len(rebound_ids) != len(rebind_ids):
        raise ValueError("重挂 fact_id 列表不合法")
    listed = tuple(sorted(rebound_ids))
    old_rows_for_ids = tuple(old_facts.require(fact_id) for fact_id in listed)
    if canonical_digest([fact.as_dict() for fact in old_rows_for_ids]) != plan_raw.get("rebind_facts_digest"):
        raise ValueError("重挂计划绑定的重挂事实摘要不符")
    unrebound = plan_raw.get("unrebound_fact_ids")
    if not isinstance(unrebound, (list, tuple)) or set(unrebound) != set(old_facts.by_id) - rebound_ids:
        raise ValueError("重挂计划的未重挂事实清单不符")

    computed = plan_rebind(base, new, old_facts, listed)
    if computed.plan.digest != plan_raw.get("digest"):
        raise ValueError("重挂计划摘要不符")
    if computed.rebound_facts.as_dicts() != facts.as_dicts():
        raise ValueError("提交事实与重挂计划结果不符（可能新桥旧引用混用）")
    if computed.deltas_as_dicts() != list(deltas_raw):
        raise ValueError("重挂差额与按新层级重算结果不符")

    return {
        "warehouse_version": warehouse_version,
        "digest": document.get("digest"),
        "hierarchy_versions": sorted(hierarchies),
        "facts": len(facts),
        "rebind": {
            "old_version": old_version,
            "new_version": new_version,
            "plan_digest": plan_raw.get("digest"),
            "rebound_fact_count": len(computed.deltas),
            "unrebound_fact_count": len(computed.plan.unrebound_fact_ids),
            "deltas": computed.deltas_as_dicts(),
        },
    }


def load_warehouse(path: str | os.PathLike[str]) -> tuple[dict, dict]:
    raw = Path(path).read_bytes()
    try:
        document = json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("仓版本不是有效UTF-8 JSON") from exc
    summary = verify_warehouse(document)
    return document, summary


def commit_warehouse(
    path: str | os.PathLike[str],
    warehouse_version: str,
    hierarchy: Hierarchy,
    facts: FactSet,
    *,
    base_hierarchy: Hierarchy | None = None,
    rebind: RebindResult | None = None,
    fault: str | None = None,
) -> dict:
    """把层级、桥、事实、重挂计划和差额作为一个文档原子提交。

    写入先落到同目录临时文件并 fsync，最后单次 ``os.replace``。任何
    异常都不会改动已有目标文件，也不会留下新桥旧事实的半成品。
    """
    document = build_warehouse_document(
        warehouse_version, hierarchy, facts,
        base_hierarchy=base_hierarchy, rebind=rebind,
    )
    payload = (canonical_json(document) + "\n").encode("utf8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        if fault == "after_staging":
            raise RuntimeError("注入故障：临时文件写入后、原子替换前失败")
        os.replace(temp_path, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        handle.close()
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    _, summary = load_warehouse(target)
    return summary
