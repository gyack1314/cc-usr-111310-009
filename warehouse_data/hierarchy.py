"""多父有向无环组织层级、精确有理权重和带路径证据的祖先桥。

方向固定为 ``child -> parent``。每个非根节点的出边权重之和必须精确等于
``1``；零权重边保留在结构中并出现在路径证据里，但对贡献只乘以零。
祖先权重是所有有向路径乘积之和，同一祖先经多条路径到达时必须相加。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

_WEIGHT_RE = re.compile(r"^[0-9]+(?:/[0-9]+)?$")


def parse_weight(value: object, what: str = "权重") -> Fraction:
    """解析严格非负有理数字符串，绝不经过浮点数。"""
    if type(value) is not str or not _WEIGHT_RE.fullmatch(value):
        raise ValueError(f"{what}必须是非负有理数字符串")
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        if int(denominator) == 0:
            raise ValueError(f"{what}分母不能为零")
        result = Fraction(int(numerator), int(denominator))
    else:
        result = Fraction(int(value))
    if result < 0:
        raise ValueError(f"{what}不能为负")
    return result


def format_weight(value: Fraction | int) -> str:
    """把精确有理数格式化为规范化字符串。"""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def parse_plain_integer(value: object, what: str = "整数") -> int:
    """只接受真正的 int；bool 和 float 都不算定标整数。"""
    if type(value) is not int:
        raise ValueError(f"{what}必须是整数")
    return value


@dataclass(frozen=True, slots=True)
class Edge:
    child: str
    parent: str
    weight: Fraction

    def as_dict(self) -> dict:
        return {
            "child": self.child,
            "parent": self.parent,
            "weight": format_weight(self.weight),
        }


@dataclass(frozen=True, slots=True)
class BridgePath:
    """从 source 到 ancestor 的一条具体路径及路径乘积。"""

    nodes: tuple[str, ...]
    weight: Fraction

    def as_dict(self) -> dict:
        return {"nodes": list(self.nodes), "weight": format_weight(self.weight)}


@dataclass(frozen=True, slots=True)
class BridgeEntry:
    """桥的一行：source 到 ancestor 的全部路径贡献之和。"""

    source: str
    ancestor: str
    weight: Fraction
    paths: tuple[BridgePath, ...]
    computed_in: str

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "ancestor": self.ancestor,
            "weight": format_weight(self.weight),
            "paths": [p.as_dict() for p in self.paths],
            "computed_in": self.computed_in,
        }


def _validate_identity(nodes: object) -> tuple[str, ...]:
    if not isinstance(nodes, (list, tuple)) or not nodes:
        raise ValueError("节点清单必须是非空数组")
    result = []
    seen = set()
    for node in nodes:
        if not isinstance(node, str) or not node:
            raise ValueError("节点身份必须是非空字符串")
        if node in seen:
            raise ValueError(f"节点重复: {node}")
        seen.add(node)
        result.append(node)
    return tuple(result)


def _validate_edges(
    nodes: Sequence[str],
    raw_edges: object,
) -> tuple[Edge, ...]:
    node_set = set(nodes)
    if not isinstance(raw_edges, (list, tuple)):
        raise ValueError("边清单必须是数组")
    edges: list[Edge] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(raw_edges):
        if not isinstance(row, Mapping):
            raise ValueError(f"第 {index} 条边必须是对象")
        child = row.get("child")
        parent = row.get("parent")
        if not isinstance(child, str) or not isinstance(parent, str):
            raise ValueError(f"第 {index} 条边缺少字符串 child/parent")
        if child not in node_set or parent not in node_set:
            raise ValueError(f"边节点未知: {child} -> {parent}")
        pair = (child, parent)
        if pair in seen_pairs:
            raise ValueError(f"重复边: {child} -> {parent}")
        seen_pairs.add(pair)
        if child == parent:
            raise ValueError(f"自环边: {child}")
        weight = parse_weight(row.get("weight"), f"边 {child} -> {parent} 的权重")
        edges.append(Edge(child, parent, weight))
    edges.sort(key=lambda e: (e.child, e.parent))
    return tuple(edges)


def _group_parents(edges: Iterable[Edge]) -> dict[str, tuple[Edge, ...]]:
    parents: dict[str, list[Edge]] = {}
    for edge in edges:
        parents.setdefault(edge.child, []).append(edge)
    return {
        child: tuple(sorted(rows, key=lambda e: e.parent))
        for child, rows in parents.items()
    }


def _validate_weights(nodes: Sequence[str], edges: tuple[Edge, ...]) -> dict[str, tuple[Edge, ...]]:
    parents = _group_parents(edges)
    for node in nodes:
        rows = parents.get(node, ())
        if not rows:
            continue  # 无父边即根
        total = sum((edge.weight for edge in rows), Fraction(0))
        if total != 1:
            raise ValueError(
                f"节点 {node} 的父权重和必须精确为1，实际为 {format_weight(total)}"
            )
    return parents


def _validate_acyclic(nodes: Sequence[str], parents: Mapping[str, Sequence[Edge]]) -> None:
    # 零权重边仍参与结构环检测。
    color: dict[str, int] = {}

    def visit(node: str, stack: tuple[str, ...]) -> None:
        state = color.get(node, 0)
        if state == 1:
            cycle = " -> ".join((*stack, node))
            raise ValueError(f"层级存在环: {cycle}")
        if state == 2:
            return
        color[node] = 1
        for edge in parents.get(node, ()):
            visit(edge.parent, (*stack, node))
        color[node] = 2

    for node in nodes:
        if color.get(node, 0) == 0:
            visit(node, ())


def _roots(nodes: Sequence[str], parents: Mapping[str, Sequence[Edge]]) -> tuple[str, ...]:
    return tuple(node for node in nodes if not parents.get(node))


def _children_map(
    nodes: Sequence[str], parents: Mapping[str, Sequence[Edge]]
) -> dict[str, tuple[str, ...]]:
    children: dict[str, list[str]] = {node: [] for node in nodes}
    for child, rows in parents.items():
        for edge in rows:
            children[edge.parent].append(child)
    return {
        node: tuple(sorted(set(rows))) for node, rows in children.items()
    }


def _topological_root_first(
    nodes: Sequence[str], parents: Mapping[str, Sequence[Edge]]
) -> tuple[str, ...]:
    """根在前、子节点在后的拓扑序；用于先算父桥再算子桥。"""
    children = _children_map(nodes, parents)
    pending = {node: len(parents.get(node, ())) for node in nodes}
    queue = deque(sorted(node for node in nodes if pending[node] == 0))
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children.get(node, ()):
            pending[child] -= 1
            if pending[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        raise ValueError("层级不是有限DAG")
    return tuple(order)


def _build_bridge_row(
    node: str,
    parent_rows: Mapping[str, Mapping[str, BridgeEntry]],
    edges: Sequence[Edge],
    computed_in: str,
) -> dict[str, BridgeEntry]:
    """父桥已就绪时，用动态规划合并当前节点的全部父路径。"""
    self_entry = BridgeEntry(
        source=node,
        ancestor=node,
        weight=Fraction(1),
        paths=(BridgePath((node,), Fraction(1)),),
        computed_in=computed_in,
    )
    merged: dict[str, tuple[Fraction, list[BridgePath]]] = {}
    for edge in edges:
        parent_row = parent_rows[edge.parent]
        for ancestor, entry in parent_row.items():
            weight = edge.weight * entry.weight
            if ancestor not in merged:
                merged[ancestor] = (Fraction(0), [])
            old_weight, old_paths = merged[ancestor]
            merged[ancestor] = (
                old_weight + weight,
                old_paths
                + [
                    BridgePath((node, *path.nodes), edge.weight * path.weight)
                    for path in entry.paths
                ],
            )
    row: dict[str, BridgeEntry] = {node: self_entry}
    for ancestor, (weight, paths) in merged.items():
        paths.sort(key=lambda p: p.nodes)
        if sum((p.weight for p in paths), Fraction(0)) != weight:
            raise ValueError(f"内部路径证据与桥权重不一致: {node} -> {ancestor}")
        row[ancestor] = BridgeEntry(
            source=node,
            ancestor=ancestor,
            weight=weight,
            paths=tuple(paths),
            computed_in=computed_in,
        )
    return row


def _build_all_rows(
    nodes: Sequence[str],
    parents: Mapping[str, Sequence[Edge]],
    version: str,
) -> dict[str, dict[str, BridgeEntry]]:
    rows: dict[str, dict[str, BridgeEntry]] = {}
    for node in _topological_root_first(nodes, parents):
        rows[node] = _build_bridge_row(
            node,
            {edge.parent: rows[edge.parent] for edge in parents.get(node, ())},
            parents.get(node, ()),
            version,
        )
    return rows


def _merge_edge_updates(
    base: "Hierarchy",
    edge_updates: Sequence[Mapping[str, object]],
) -> tuple[tuple[Edge, ...], dict[tuple[str, str], Fraction]]:
    if not isinstance(edge_updates, (list, tuple)):
        raise ValueError("改边清单必须是数组")
    update_map: dict[tuple[str, str], Fraction] = {}
    for index, row in enumerate(edge_updates):
        if not isinstance(row, Mapping):
            raise ValueError(f"第 {index} 条改边必须是对象")
        child = row.get("child")
        parent = row.get("parent")
        if not isinstance(child, str) or not isinstance(parent, str):
            raise ValueError(f"第 {index} 条改边缺少字符串 child/parent")
        pair = (child, parent)
        if pair in update_map:
            raise ValueError(f"重复改边: {child} -> {parent}")
        if child not in set(base.nodes) or parent not in set(base.nodes):
            raise ValueError(f"改边节点未知: {child} -> {parent}")
        if not any(
            edge.child == child and edge.parent == parent for edge in base.edges
        ):
            raise ValueError(f"改边只能修改既有边，未知边: {child} -> {parent}")
        update_map[pair] = parse_weight(
            row.get("weight"), f"改边 {child} -> {parent} 的权重"
        )
    new_edges = tuple(
        Edge(
            edge.child,
            edge.parent,
            update_map.get((edge.child, edge.parent), edge.weight),
        )
        for edge in base.edges
    )
    return new_edges, update_map


def _load_supplied_bridge(
    nodes: Sequence[str],
    parents: Mapping[str, Sequence[Edge]],
    version: str,
    supplied: object,
) -> dict[str, dict[str, BridgeEntry]]:
    """装载已提交的桥行，并逐行校验权重与路径证据。

    已提交的新版本桥允许保留 ``computed_in`` 指向父版本，用于说明该行
    没有在本次版本重算；其权重与路径必须仍与当前边一致。
    """
    if not isinstance(supplied, Mapping):
        raise ValueError("已提交桥必须是对象")
    rows: dict[str, dict[str, BridgeEntry]] = {}
    for source in nodes:
        raw_row = supplied.get(source)
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"已提交桥缺少节点 {source}")
        row: dict[str, BridgeEntry] = {}
        for ancestor, raw_entry in raw_row.items():
            if not isinstance(ancestor, str) or ancestor not in set(nodes):
                raise ValueError(f"已提交桥引用未知祖先 {ancestor}")
            if not isinstance(raw_entry, Mapping):
                raise ValueError("桥条目必须是对象")
            weight = parse_weight(raw_entry.get("weight"), "已提交桥权重")
            computed_in = raw_entry.get("computed_in")
            if not isinstance(computed_in, str) or not computed_in:
                raise ValueError("桥条目缺少 computed_in")
            raw_paths = raw_entry.get("paths")
            if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
                raise ValueError("桥条目缺少路径证据")
            paths: list[BridgePath] = []
            for raw_path in raw_paths:
                if not isinstance(raw_path, Mapping):
                    raise ValueError("路径证据必须是对象")
                raw_nodes = raw_path.get("nodes")
                if not isinstance(raw_nodes, (list, tuple)) or not raw_nodes:
                    raise ValueError("路径证据缺少节点序列")
                if not all(isinstance(item, str) and item in set(nodes) for item in raw_nodes):
                    raise ValueError("路径证据引用未知节点")
                path_weight = parse_weight(raw_path.get("weight"), "路径权重")
                paths.append(BridgePath(tuple(raw_nodes), path_weight))
            paths.sort(key=lambda p: p.nodes)
            row[ancestor] = BridgeEntry(source, ancestor, weight, tuple(paths), computed_in)
        self_expected = BridgePath((source,), Fraction(1))
        self_entry = row.get(source)
        if self_entry is None or self_entry.weight != 1 or self_entry.paths != (self_expected,):
            raise ValueError(f"节点 {source} 的自桥必须是自身权重1")
        rows[source] = row

    # 按当前边独立重算一份，逐祖先对照；任何新旧桥混用都会在这里暴露。
    expected_rows = _build_all_rows(nodes, parents, version)
    for source in nodes:
        actual = {ancestor: entry for ancestor, entry in rows[source].items()}
        expected = expected_rows[source]
        if set(actual) != set(expected):
            raise ValueError(f"节点 {source} 的已提交桥祖先集合与边不一致")
        for ancestor, expected_entry in expected.items():
            actual_entry = actual[ancestor]
            if actual_entry.weight != expected_entry.weight:
                raise ValueError(f"桥权重与边不一致: {source} -> {ancestor}")
            actual_paths = sorted(
                (path.nodes, path.weight) for path in actual_entry.paths
            )
            expected_paths = sorted(
                (path.nodes, path.weight) for path in expected_entry.paths
            )
            if actual_paths != expected_paths:
                raise ValueError(f"桥路径证据与边不一致: {source} -> {ancestor}")
    return rows


def _root_conservation(
    nodes: Sequence[str],
    roots: Sequence[str],
    rows: Mapping[str, Mapping[str, BridgeEntry]],
) -> dict[str, Fraction]:
    result: dict[str, Fraction] = {}
    for node in nodes:
        total = sum(
            (rows[node][root].weight for root in roots if root in rows[node]),
            Fraction(0),
        )
        if total != 1:
            raise ValueError(
                f"节点 {node} 流向全部根的权重和必须为1，实际为 {format_weight(total)}"
            )
        result[node] = total
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class Hierarchy:
    """一个不可变的版本化层级及其完整祖先桥。"""

    version: str
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    parents: Mapping[str, tuple[Edge, ...]]
    roots: tuple[str, ...]
    rows: Mapping[str, Mapping[str, BridgeEntry]]
    parent_version: str | None = None
    changed_children: tuple[str, ...] = ()
    recomputed_nodes: tuple[str, ...] = ()
    reused_nodes: tuple[str, ...] = ()
    root_totals: Mapping[str, Fraction] | None = None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, version: str | None = None
    ) -> "Hierarchy":
        if not isinstance(value, Mapping):
            raise ValueError("层级必须是对象")
        raw_version = value.get("version") if version is None else version
        if not isinstance(raw_version, str) or not raw_version:
            raise ValueError("层级版本必须是非空字符串")
        parent_version = value.get("parent_version")
        if parent_version is not None and (
            not isinstance(parent_version, str) or not parent_version
        ):
            raise ValueError("父层级版本必须是非空字符串或 null")
        nodes = _validate_identity(value.get("nodes"))
        edges = _validate_edges(nodes, value.get("edges"))
        parents = _validate_weights(nodes, edges)
        _validate_acyclic(nodes, parents)
        roots = _roots(nodes, parents)
        summary = value.get("summary")
        raw_changed: object = ()
        raw_recomputed: object = ()
        raw_reused: object = ()
        if isinstance(summary, Mapping):
            raw_changed = summary.get("changed_children", ())
            raw_recomputed = summary.get("recomputed_nodes", ())
            raw_reused = summary.get("reused_nodes", ())
        for field, raw in (
            ("changed_children", raw_changed),
            ("recomputed_nodes", raw_recomputed),
            ("reused_nodes", raw_reused),
        ):
            if not isinstance(raw, (list, tuple)) or not all(
                isinstance(item, str) and item in set(nodes) for item in raw
            ):
                raise ValueError(f"层级摘要的 {field} 不合法")
        changed_children = tuple(sorted(raw_changed))
        recomputed_nodes = tuple(sorted(raw_recomputed))
        reused_nodes = tuple(sorted(raw_reused))
        if set(recomputed_nodes) & set(reused_nodes):
            raise ValueError("层级摘要的重算与复用节点不能重叠")
        if parent_version is None:
            if recomputed_nodes or reused_nodes or changed_children:
                raise ValueError("基线层级不应声明重算或复用节点")
        elif set(recomputed_nodes) | set(reused_nodes) != set(nodes):
            raise ValueError("层级摘要必须完整划分重算与复用节点")

        # 优先按已提交的桥证据装载；没有证据时按边重算，并要求两者一致。
        supplied_bridge = value.get("bridge")
        if supplied_bridge is None:
            rows = _build_all_rows(nodes, parents, raw_version)
        else:
            rows = _load_supplied_bridge(nodes, parents, raw_version, supplied_bridge)
            # 重算来源必须与摘要声明的闭包一致：本版本重算的节点，其每行都
            # 由本版本生成；复用的节点不得出现本版本生成的任何行，否则就
            # 是用新版本号冒充局部重算，或把旧行错标成新桥。
            if parent_version is not None:
                for node in recomputed_nodes:
                    for entry in rows[node].values():
                        if entry.computed_in != raw_version:
                            raise ValueError(f"重算节点 {node} 的桥来源不是 {raw_version}")
                for node in reused_nodes:
                    for entry in rows[node].values():
                        if entry.computed_in == raw_version:
                            raise ValueError(f"复用节点 {node} 的桥不应标记为本版本重算")
        totals = _root_conservation(nodes, roots, rows)
        return cls(
            version=raw_version,
            nodes=nodes,
            edges=edges,
            parents=parents,
            roots=roots,
            rows=rows,
            parent_version=parent_version,
            changed_children=changed_children,
            recomputed_nodes=recomputed_nodes,
            reused_nodes=reused_nodes,
            root_totals=totals,
        )

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> "Hierarchy":
        if not isinstance(manifest, Mapping):
            raise ValueError("清单必须是对象")
        kind = manifest.get("kind")
        if kind is not None and kind != "weighted-hierarchy":
            raise ValueError("不支持的资料类型")
        return cls.from_mapping(manifest)

    def derive_parent_edges(self, edge_updates: Sequence[Mapping[str, object]]) -> tuple[Edge, ...]:
        """校验一组局部改边，返回合并后的完整边集（不构建新层级）。"""
        return _merge_edge_updates(self, edge_updates)[0]

    @classmethod
    def successor(
        cls,
        base: "Hierarchy",
        version: str,
        edge_updates: Sequence[Mapping[str, object]],
    ) -> "Hierarchy":
        """按局部改权创建新版本，只重算改边子节点及其反向后代闭包。"""
        if not isinstance(version, str) or not version:
            raise ValueError("新层级版本必须是非空字符串")
        if version == base.version:
            raise ValueError("新层级版本必须不同于旧版本")
        new_edges, update_map = _merge_edge_updates(base, edge_updates)
        parents = _validate_weights(base.nodes, new_edges)
        _validate_acyclic(base.nodes, parents)

        changed_children = {
            edge.child
            for edge in base.edges
            if update_map.get((edge.child, edge.parent), edge.weight) != edge.weight
        }
        if not changed_children:
            raise ValueError("没有任何实际权重变化，不应创建新层级版本")
        # 反向边（父 -> 子）闭包；含改边子节点自身。
        children = _children_map(base.nodes, parents)
        closure: set[str] = set()
        stack = list(changed_children)
        while stack:
            node = stack.pop()
            if node in closure:
                continue
            closure.add(node)
            stack.extend(children.get(node, ()))

        rows: dict[str, dict[str, BridgeEntry]] = {}
        for node in _topological_root_first(base.nodes, parents):
            if node in closure:
                rows[node] = _build_bridge_row(
                    node,
                    {edge.parent: rows[edge.parent] for edge in parents.get(node, ())},
                    parents.get(node, ()),
                    version,
                )
            else:
                # 未在反向后代闭包中的节点不重算，原样携带旧版本路径证据。
                rows[node] = base.rows[node]

        roots = _roots(base.nodes, parents)
        totals = _root_conservation(base.nodes, roots, rows)
        return cls(
            version=version,
            nodes=base.nodes,
            edges=new_edges,
            parents=parents,
            roots=roots,
            rows=rows,
            parent_version=base.version,
            changed_children=tuple(sorted(changed_children)),
            recomputed_nodes=tuple(sorted(closure)),
            reused_nodes=tuple(sorted(set(base.nodes) - closure)),
            root_totals=totals,
        )

    # ---- 查询 ----
    def is_root(self, node: str) -> bool:
        self._require_node(node)
        return node in self.roots

    def parent_edges(self, node: str) -> tuple[Edge, ...]:
        self._require_node(node)
        return self.parents.get(node, ())

    def ancestors_of(self, node: str) -> tuple[str, ...]:
        self._require_node(node)
        return tuple(self.rows[node])

    def bridge_entry(self, source: str, ancestor: str) -> BridgeEntry | None:
        self._require_node(source)
        self._require_node(ancestor)
        return self.rows[source].get(ancestor)

    def bridge_weight(self, source: str, ancestor: str) -> Fraction:
        entry = self.bridge_entry(source, ancestor)
        return Fraction(0) if entry is None else entry.weight

    def is_ancestor(self, ancestor: str, descendant: str, *, strict: bool = True) -> bool:
        """ancestor 是否在 descendant 的上卷路径上。"""
        self._require_node(ancestor)
        self._require_node(descendant)
        if ancestor == descendant:
            return not strict
        return ancestor in self.rows[descendant]

    def closure_of_changed_children(self, changed_children: Iterable[str]) -> tuple[str, ...]:
        """给出改权子节点及其全部反向后代（子方向），不作树化修正。

        这是独立的闭包枚举：沿父 -> 子边行走，多父节点只进入一次但不截断
        其它分支。``successor`` 用同一规则决定实际重算范围。
        """
        nodes_set = set(self.nodes)
        requested = [node for node in changed_children if node in nodes_set]
        unknown = [node for node in changed_children if node not in nodes_set]
        if unknown:
            raise ValueError(f"闭包起点未知节点: {unknown}")
        children = _children_map(self.nodes, self.parents)
        closure: set[str] = set()
        stack = list(requested)
        while stack:
            node = stack.pop()
            if node in closure:
                continue
            closure.add(node)
            stack.extend(children.get(node, ()))
        return tuple(sorted(closure))

    def _require_node(self, node: str) -> None:
        if node not in set(self.nodes):
            raise ValueError(f"未知节点: {node}")

    # ---- 校验、摘要与序列化 ----
    def verify_bridge(self) -> dict:
        """独立枚举全部简单路径，逐祖先对照动态规划桥和路径证据。"""
        parent_map = {
            node: tuple(
                (edge.parent, edge.weight) for edge in self.parents.get(node, ())
            )
            for node in self.nodes
        }
        report: dict = {
            "version": self.version,
            "entries": 0,
            "paths": 0,
            "independent_enumeration": True,
            "ok": True,
        }
        for source in self.nodes:
            enumerated: dict[str, dict[tuple[str, ...], Fraction]] = {}
            stack: list[tuple[str, tuple[str, ...], Fraction]] = [
                (source, (source,), Fraction(1))
            ]
            while stack:
                node, path, product = stack.pop()
                if len(path) > 1:
                    enumerated.setdefault(node, {})[path] = product
                for parent, weight in parent_map[node]:
                    if parent in path:
                        raise ValueError(f"路径枚举发现环: {' -> '.join((*path, parent))}")
                    stack.append((parent, (*path, parent), product * weight))
            for ancestor, entry in self.rows[source].items():
                report["entries"] += 1
                if ancestor == source:
                    expected_paths = {(source,): Fraction(1)}
                else:
                    expected_paths = enumerated.get(ancestor, {})
                evidence = {path.nodes: path.weight for path in entry.paths}
                report["paths"] += len(evidence)
                enumerated_sum = sum(expected_paths.values(), Fraction(0))
                evidence_sum = sum(evidence.values(), Fraction(0))
                if (
                    len(evidence) != len(expected_paths)
                    or evidence != expected_paths
                    or entry.weight != enumerated_sum
                    or evidence_sum != entry.weight
                ):
                    report["ok"] = False
                    raise ValueError(
                        f"桥与独立路径枚举不一致: {source} -> {ancestor}"
                    )
        return report

    @property
    def digest(self) -> str:
        canonical = {
            "version": self.version,
            "parent_version": self.parent_version,
            "nodes": list(self.nodes),
            "edges": [edge.as_dict() for edge in self.edges],
            "roots": list(self.roots),
            "bridge": {
                source: {
                    ancestor: {
                        "weight": format_weight(entry.weight),
                        "paths": [path.as_dict() for path in entry.paths],
                        "computed_in": entry.computed_in,
                    }
                    for ancestor, entry in sorted(row.items())
                }
                for source, row in sorted(self.rows.items())
            },
        }
        return hashlib.sha256(_canonical_json(canonical).encode("utf8")).hexdigest()

    def summary(self) -> dict:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "digest": self.digest,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "roots": list(self.roots),
            "bridge_entries": sum(len(row) for row in self.rows.values()),
            "changed_children": list(self.changed_children),
            "recomputed_nodes": list(self.recomputed_nodes),
            "reused_nodes": list(self.reused_nodes),
        }

    def bridge_as_dict(self) -> dict:
        return {
            source: {
                ancestor: entry.as_dict()
                for ancestor, entry in sorted(row.items())
            }
            for source, row in sorted(self.rows.items())
        }

    def edge_as_dicts(self) -> list[dict]:
        return [edge.as_dict() for edge in self.edges]

    def parent_counts(self) -> dict[str, int]:
        return {node: len(self.parents.get(node, ())) for node in self.nodes}
