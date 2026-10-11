"""在本机的测试视频上跑一遍，和人工核对过的结果比。视频不在就跳过。

python -m video2score.bench                 全部
python -m video2score.bench <名字>     只跑一个（tests/golden/<名字>.json）

比两样：小节格数；读出的小节号序列和印在谱上的序列（缺号、重号、顺序错）。
"""

from __future__ import annotations

import json
import sys
import time

from video2score.jobs import ROOT, load_manifest
from video2score.run import Options, run

GOLDEN = ROOT / "tests" / "golden"


def parse_labels(spec: str) -> list[int]:
    """"1,3-62,63" → [1, 3, 4, …, 62, 63]。"""
    numbers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(value) for value in part.split("-", 1))
            numbers.extend(range(first, last + 1))
        else:
            numbers.append(int(part))
    return numbers


def compare_labels(expected: list[int], got: list[int | None]) -> dict[str, object]:
    """读出的号（按输出顺序，读不出为 None）和期望序列比。"""
    read = [number for number in got if number is not None]
    expected_set = set(expected)
    seen: dict[int, int] = {}
    for number in read:
        seen[number] = seen.get(number, 0) + 1
    common = _lcs_length(expected, read)
    return {
        "coverage": len(read) / max(1, len(got)),
        "in_order": common / max(1, len(expected)),
        "missing": [number for number in expected if number not in seen],
        "duplicates": sorted(number for number, count in seen.items() if count > 1),
        "unexpected": sorted(number for number in seen if number not in expected_set),
        "out_of_order": len([n for n in read if n in expected_set]) - common,
    }


def _lcs_length(a: list[int], b: list[int]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def main(argv: list[str] | None = None) -> int:
    wanted = set(argv if argv is not None else sys.argv[1:])
    failed = 0
    rows: list[str] = []
    for path in sorted(GOLDEN.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem
        if wanted and name not in wanted:
            continue
        video = ROOT / case["video"]
        if not video.is_file():
            rows.append(f"{name:14s} 跳过（没有 {case['video']}）")
            continue
        started = time.time()
        job = run(str(video), Options(debug=True))
        elapsed = time.time() - started
        got = int(load_manifest(job)["measures"])
        expected = int(case["measures"])
        ok = abs(got - expected) <= int(case.get("tolerance", 0))
        line = f"{name:14s} {'通过' if ok else '不符'}  格数 {got}/{expected}  {elapsed:4.0f} 秒  {job}"
        if case.get("labels"):
            library = json.loads((job / "measures.json").read_text(encoding="utf-8"))
            report = compare_labels(parse_labels(case["labels"]), [item.get("number") for item in library])
            line += (
                f"\n{'':16s}读号 {report['coverage']:.0%}  顺序吻合 {report['in_order']:.1%}  "
                f"错序 {report['out_of_order']}  缺号 {_short(report['missing'])}  "
                f"重号 {_short(report['duplicates'])}  多余 {_short(report['unexpected'])}"
            )
        failed += 0 if ok else 1
        rows.append(line)
        print(line, flush=True)
    print("\n".join(["", *rows]))
    return 1 if failed else 0


def _short(values: object) -> str:
    items = list(values) if isinstance(values, list) else []
    if not items:
        return "无"
    text = ",".join(str(item) for item in items[:12])
    return f"{len(items)} 个[{text}{'…' if len(items) > 12 else ''}]"


if __name__ == "__main__":
    raise SystemExit(main())
