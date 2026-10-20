"""Day32 第2步：SBOM依赖风险扫描。

python day32/python/day32_dependency_scan.py
python day32/python/day32_dependency_scan.py --mode online

退出码：0=查询完成且未发现条目，1=发现条目，2=未知或查询不完整。
demo模式使用虚构规则，只用于练习，不是实际漏洞结论。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from day32_sbom_basic import write_json_atomically


DAY32_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SBOM = DAY32_DIR / "reports/rag-service-31.0.0.sbom.json"
OSV_URL = "https://api.osv.dev/v1/querybatch"


def load_dependencies(path: Path) -> tuple[dict, list[dict]]:
    sbom = json.loads(path.read_text(encoding="utf-8"))
    if sbom.get("format") != "day32-teaching-sbom":
        raise ValueError("输入必须是第1步生成的教学SBOM。")
    dependencies = sbom["dependencies"]
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("依赖列表不能为空。")
    for package in dependencies:
        if package.get("ecosystem") != "PyPI":
            raise ValueError("当前只支持PyPI依赖。")
        for field in ("name", "version"):
            if not isinstance(package.get(field), str) or not package[field]:
                raise ValueError(f"依赖缺少有效字段：{field}")
    return sbom, dependencies


def demo_scan(dependencies: list[dict]) -> list[dict]:
    """故意模拟三个状态，不把虚构规则伪装成CVE/GHSA。"""
    results = []
    for index, package in enumerate(dependencies):
        if index == 0:
            state = "findings"
            findings = [{"id": "DEMO-ONLY-001", "description": "虚构教学风险，无实际漏洞含义。"}]
        elif index == 1:
            state, findings = "no_known_findings", []
        else:
            state, findings = "unknown", []
        results.append({
            "name": package["name"], "version": package["version"],
            "status": state, "findings": findings,
            "source": "fictional-demo-rules",
        })
    return results


def post_osv(queries: list[dict]) -> list[dict]:
    request = Request(
        OSV_URL,
        data=json.dumps({"queries": queries}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "day32-learning/1"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        body = json.load(response)
    results = body.get("results")
    if not isinstance(results, list) or len(results) != len(queries):
        raise ValueError("OSV返回的结果数与查询数不一致。")
    return results


def online_scan(dependencies: list[dict]) -> list[dict]:
    """查询精确版本；分页失败时保留已发现条目并标为unknown。"""
    results = [{
        "name": item["name"], "version": item["version"],
        "status": "unknown", "findings": [], "source": "OSV",
    } for item in dependencies]
    pending = [(index, {
        "package": {"name": item["name"], "ecosystem": "PyPI"},
        "version": item["version"],
    }) for index, item in enumerate(dependencies)]

    try:
        # 上限避免异常上游无限分页。没有完整查完就保持unknown。
        for _ in range(20):
            if not pending:
                break
            batch = post_osv([query for _, query in pending])
            next_pending = []
            for (index, query), response in zip(pending, batch):
                if not isinstance(response, dict):
                    raise ValueError("OSV结果必须是对象。")
                findings = response.get("vulns", [])
                if not isinstance(findings, list) or any(
                    not isinstance(item, dict) or not isinstance(item.get("id"), str)
                    for item in findings
                ):
                    raise ValueError("OSV漏洞条目格式无效。")
                results[index]["findings"].extend(findings)
                token = response.get("next_page_token")
                if token:
                    next_pending.append((index, {**query, "page_token": token}))
                else:
                    results[index]["status"] = (
                        "findings" if results[index]["findings"] else "no_known_findings"
                    )
            pending = next_pending
        if pending:
            for index, _ in pending:
                results[index]["error"] = "超过分页上限，查询未完成。"
    except Exception as error:
        for result in results:
            if result["status"] == "unknown":
                result["error"] = f"{type(error).__name__}: {error}"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Day32依赖风险扫描")
    parser.add_argument("--sbom", type=Path, default=DEFAULT_SBOM)
    parser.add_argument("--mode", choices=("demo", "online"), default="demo")
    args = parser.parse_args()
    sbom, dependencies = load_dependencies(args.sbom)
    results = demo_scan(dependencies) if args.mode == "demo" else online_scan(dependencies)
    unknown = sum(item["status"] == "unknown" for item in results)
    finding_packages = sum(bool(item["findings"]) for item in results)
    exit_code = 2 if unknown else (1 if finding_packages else 0)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": args.mode, "is_demo": args.mode == "demo",
        "artifact": sbom["artifact"],
        "coverage": "declared_direct_dependencies_only",
        "summary": {
            "packages": len(results), "packages_with_findings": finding_packages,
            "unknown": unknown, "exit_code": exit_code,
            "decision": "blocked" if exit_code else "no_known_findings",
        },
        "results": results,
    }
    output = DAY32_DIR / "reports" / f"day32-dependency-scan-{args.mode}.json"
    write_json_atomically(output, report)
    print("Day32 依赖版本与风险扫描")
    print("模式：", "离线虚构演示" if args.mode == "demo" else "真实OSV查询")
    for item in results:
        print(f"- {item['name']}=={item['version']} | {item['status']}")
        for finding in item["findings"]:
            print("  条目：", finding["id"])
        if "error" in item:
            print("  原因：", item["error"])
    print("\n汇总：", report["summary"])
    print("报告：", output)
    print("\n未知或网络失败不能判为安全；查无记录也不代表没有漏洞。")
    print("当前仅扫描SBOM中的直接依赖。")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
