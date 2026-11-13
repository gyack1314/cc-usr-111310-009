"""查看层级桥、事实分摊、汇总选择，并以原子方式提交仓版本。"""
import argparse
import json
import sys

from .facts import (
    FactSet,
    RebindResult,
    allocate_facts,
    assess_rollup,
    commit_warehouse,
    load_warehouse,
    plan_rebind,
)
from .hierarchy import Hierarchy
from .io import load


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _load_hierarchy_and_facts(path: str) -> tuple[Hierarchy, FactSet, dict]:
    manifest, digest = load(path)
    hierarchy = Hierarchy.from_manifest(manifest)
    facts = FactSet.from_manifest(manifest, hierarchy=hierarchy)
    return hierarchy, facts, {"manifest_sha256": digest, **hierarchy.summary()}


def cmd_inspect(args) -> int:
    manifest, digest = load(args.file)
    hierarchy = Hierarchy.from_manifest(manifest)
    facts = FactSet.from_manifest(manifest, hierarchy=hierarchy)
    _print({
        "manifest_sha256": digest,
        "hierarchy": hierarchy.summary(),
        "bridge_check": hierarchy.verify_bridge(),
        "allocation": allocate_facts(facts, hierarchy, include_paths=args.paths),
    })
    return 0


def cmd_bridge(args) -> int:
    hierarchy, facts, meta = _load_hierarchy_and_facts(args.file)
    bridge_report = hierarchy.verify_bridge()
    payload = {
        **meta,
        "bridge": hierarchy.bridge_as_dict(),
        "bridge_check": bridge_report,
        "facts": [(fact.fact_id, fact.node, fact.value, fact.hierarchy_version) for fact in facts],
    }
    if args.fact:
        payload["allocations"] = allocate_facts(facts, hierarchy, include_paths=True)
    _print(payload)
    return 0


def cmd_rollup(args) -> int:
    manifest, _digest = load(args.file)
    hierarchy = Hierarchy.from_manifest(manifest)
    facts = FactSet.from_manifest(manifest, hierarchy=hierarchy)
    selected = args.nodes or manifest.get("selected_rollup_nodes")
    result = assess_rollup(facts, hierarchy, selected)
    _print(result)
    return 0


def cmd_version(args) -> int:
    manifest, _digest = load(args.file)
    base = Hierarchy.from_manifest(manifest)
    facts = FactSet.from_manifest(manifest, hierarchy=base)
    new_edges = manifest.get("new_edge_weights", [])
    if not new_edges:
        raise ValueError("清单缺少 new_edge_weights，无法推出新层级版本")
    hierarchy = Hierarchy.successor(base, args.new_version, new_edges)
    rebind_rows = manifest.get("rebind_fact_ids", [])
    planned = (
        plan_rebind(base, hierarchy, facts, rebind_rows) if rebind_rows else None
    )
    _print({
        "manifest_sha256": digest,
        "base": base.summary(),
        "new": hierarchy.summary(),
        "new_bridge_check": hierarchy.verify_bridge(),
        "rebind": None if planned is None else {
            "plan": planned.plan.as_dict(),
            "deltas": planned.deltas_as_dicts(),
        },
    })
    return 0


def cmd_commit(args) -> int:
    manifest, _digest = load(args.file)
    base = Hierarchy.from_manifest(manifest)
    facts = FactSet.from_manifest(manifest, hierarchy=base)
    rebind_rows = manifest.get("rebind_fact_ids", [])
    if not args.new_version:
        # 显式重挂必须伴随新层级版本；否则不把旧绑定事实偷偷切版本。
        rebind_rows = []
    if args.new_version:
        new_edges = manifest.get("new_edge_weights", [])
        if not new_edges:
            raise ValueError("清单缺少 new_edge_weights，无法推出新层级版本")
        hierarchy = Hierarchy.successor(base, args.new_version, new_edges)
    else:
        hierarchy = base
    planned: RebindResult | None = None
    if rebind_rows:
        if hierarchy.version == base.version:
            raise ValueError("重挂事实必须有新的层级版本")
        planned = plan_rebind(base, hierarchy, facts, rebind_rows)
    summary = commit_warehouse(
        args.out,
        args.warehouse_version,
        hierarchy,
        facts,
        base_hierarchy=None if planned is None else base,
        rebind=planned,
        fault=args.fail_after_staging,
    )
    _print(summary)
    return 0


def cmd_verify(args) -> int:
    _document, summary = load_warehouse(args.file)
    _print(summary)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="warehouse_data")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="查看层级、桥自检和事实分摊")
    p.add_argument("file")
    p.add_argument("--paths", action="store_true", help="输出全部路径证据")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("bridge", help="查看完整带路径证据的桥")
    p.add_argument("file")
    p.add_argument("--fact", action="store_true", help="同时输出事实分摊")
    p.set_defaults(func=cmd_bridge)

    p = sub.add_parser("rollup", help="按指定节点评估是否可直接相加")
    p.add_argument("file")
    p.add_argument("nodes", nargs="*")
    p.set_defaults(func=cmd_rollup)

    p = sub.add_parser("version", help="推出新层级版本并预览重挂差额")
    p.add_argument("file")
    p.add_argument("--new-version", required=True)
    p.set_defaults(func=cmd_version)

    p = sub.add_parser("commit", help="原子提交仓版本")
    p.add_argument("file")
    p.add_argument("--out", required=True)
    p.add_argument("--warehouse-version", required=True)
    p.add_argument("--new-version", default=None)
    p.add_argument(
        "--fail-after-staging",
        default=None,
        const="after_staging",
        nargs="?",
        help="测试用故障注入：临时文件写入后、原子替换前失败",
    )
    p.set_defaults(func=cmd_commit)

    p = sub.add_parser("verify", help="重算并核对仓版本")
    p.add_argument("file")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
