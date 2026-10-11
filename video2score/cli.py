"""python -m video2score 的命令行。"""

from __future__ import annotations

import argparse
import sys

from video2score.jobs import load_manifest
from video2score.run import DEFAULT_SAMPLE_EVERY, Options, run


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["app"]:
        # 网页界面的参数由它自己解析（打包好的 video2score 直接调它）。
        from video2score.app.server import main as app_main

        return app_main(argv[1:])
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "glyphs":
        from video2score.glyphs import GLYPHS, build

        counts = build()
        print(f"{GLYPHS} " + " ".join(f"{name}={count}" for name, count in counts.items()))
        return 0
    if args.command == "tab":
        return _tab(args.job, _formats(args.export), args.link)
    if args.command == "export":
        return _export(args)
    if args.command == "compare":
        return _compare(args.job, args.reference)
    if args.command == "pdf":
        from video2score.jobs import resolve_job_dir
        from video2score.run import rewrite_pdf

        print(rewrite_pdf(resolve_job_dir(args.job), args.link))
        return 0
    if args.command == "bench-pdf":
        from video2score.benchpdf import main as bench_pdf

        return bench_pdf((["--rerun"] if args.rerun else []) + list(args.names))
    options = Options(
        notation=args.notation,
        start_sec=args.start or 0.0,
        end_sec=args.end,
        sample_every=args.sample_every,
        ratios=(args.crop_x_start, args.crop_y_start, args.crop_x_end, args.crop_y_end),
        debug=args.debug,
        exports=_formats(args.export),
        link=args.link,
    )
    progress = None if args.quiet else (lambda text: print(text, file=sys.stderr, flush=True))
    try:
        job_dir = run(args.source, options, progress=progress)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    manifest = load_manifest(job_dir)
    print(
        f"{job_dir} status={manifest['status']} notation={manifest['notation']} "
        f"measures={manifest['measures']} pages={manifest['pages']}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2score",
        description="把动态谱视频裁成不重复的小节，清洗后重新排版成 PDF。",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="下载或读取视频，输出 score.pdf 和小节库")
    run_parser.add_argument("--source", required=True, help="B 站链接、BV 号、YouTube 链接、其他视频网站链接或本地视频")
    run_parser.add_argument(
        "--notation",
        choices=("staff", "tab", "staff+tab"),
        default=None,
        help="谱类。默认按谱线条数自动判断",
    )
    run_parser.add_argument("--start", type=float, default=None, help="起始秒")
    run_parser.add_argument("--end", type=float, default=None, help="结束秒")
    run_parser.add_argument(
        "--sample-every",
        type=float,
        default=DEFAULT_SAMPLE_EVERY,
        help=f"取帧间隔（秒），默认 {DEFAULT_SAMPLE_EVERY}",
    )
    for name in ("x-start", "y-start", "x-end", "y-end"):
        run_parser.add_argument(f"--crop-{name}", type=float, default=None, help="谱面区域比例（0~1），不给则自动找")
    run_parser.add_argument("--debug", action="store_true", help="另存帧间对应记录和小节号拼图")
    run_parser.add_argument("--quiet", action="store_true", help="不打印进度")
    run_parser.add_argument("--export", default="", help=_EXPORT_HELP)
    run_parser.add_argument("--link", default=None, help=_LINK_HELP)
    tab_parser = commands.add_parser("tab", help="对已有作业重新读 TAB 和节奏，写 score.musicxml")
    tab_parser.add_argument("--job", required=True, help="作业 id 或作业目录")
    tab_parser.add_argument("--export", default="", help=_EXPORT_HELP)
    tab_parser.add_argument("--link", default=None, help=_LINK_HELP)
    export_parser = commands.add_parser("export", help="把识别出的 MusicXML 导出成 Guitar Pro 文件或重新排版的 PDF")
    export_from = export_parser.add_mutually_exclusive_group(required=True)
    export_from.add_argument("--job", help="作业 id 或作业目录（用它的 score.musicxml）")
    export_from.add_argument("--musicxml", help="任意一份带 TAB 行的 MusicXML（比如 GP 导出的）")
    export_parser.add_argument("--to", required=True, help="逗号分开：gp（GP7/8）、gpx（GP6）、gp5（GP5）、pdf")
    export_parser.add_argument("--out", default=None, help="输出目录（--musicxml 时默认和它放在一起）")
    compare_parser = commands.add_parser("compare", help="和 Guitar Pro 导出的 PDF 原谱逐小节比对，写 compare.txt")
    compare_parser.add_argument("--job", required=True, help="作业 id 或作业目录")
    compare_parser.add_argument("--reference", required=True, help="原谱 PDF（Guitar Pro 导出）")
    pdf_parser = commands.add_parser("pdf", help="用已有作业的小节库重新排版 score.pdf（不重跑视频）")
    pdf_parser.add_argument("--job", required=True, help="作业 id 或作业目录")
    pdf_parser.add_argument("--link", default=None, help=_LINK_HELP)
    commands.add_parser("app", help="打开本机网页界面（--port、--no-browser、--selftest）")
    commands.add_parser("glyphs", help="用本机字体重新生成冻结的字形模板和字宽（resources/glyphs.npz、font_widths.json）")
    bench_parser = commands.add_parser("bench-pdf", help="在 jobs/_user_test 里的视频 + 原谱 PDF 配对上跑基准")
    bench_parser.add_argument("names", nargs="*", help="只跑这几首（目录名）")
    bench_parser.add_argument("--rerun", action="store_true", help="重新跑视频，不复用已有作业")
    return parser


_EXPORT_HELP = "另外导出的格式，逗号分开：gp（GP7/8）、gpx（GP6）、gp5（GP5）、pdf（按识别结果重新排版的谱）"


_LINK_HELP = "原视频链接（B 站链接、BV 号、YouTube 或其他网站链接）：本地视频用它查曲名、原唱，每页页脚印上视频作者和链接"


def _formats(text: str | None) -> tuple[str, ...]:
    return tuple(item.strip().lower().lstrip(".") for item in (text or "").split(",") if item.strip())


def _export(args: argparse.Namespace) -> int:
    from pathlib import Path

    from video2score.jobs import resolve_job_dir, save_manifest
    from video2score.run import EXPORT_FORMATS, export_score

    formats = _formats(args.to)
    try:
        unknown = [item for item in formats if item not in EXPORT_FORMATS]
        if unknown or not formats:
            raise ValueError(f"不支持的导出格式：{', '.join(unknown) or '（空）'}（可选 {', '.join(EXPORT_FORMATS)}）")
        if args.job:
            job_dir = resolve_job_dir(args.job)
            manifest = load_manifest(job_dir)
            written = export_score(job_dir, manifest, formats)
            save_manifest(job_dir, manifest)
        else:
            from video2score.engrave import musicxml_to_pdf
            from video2score.gpfile import export

            source = Path(args.musicxml)
            folder = Path(args.out) if args.out else source.parent
            folder.mkdir(parents=True, exist_ok=True)
            written = export(source, list(formats), folder, source.stem)
            if "pdf" in formats:
                written.append(musicxml_to_pdf(source, folder / f"{source.stem}_engraved.pdf"))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


def _tab(job: str, exports: tuple[str, ...] = (), link: str | None = None) -> int:
    from video2score.jobs import resolve_job_dir, save_manifest
    from video2score.run import video_of, write_tab_score

    try:
        job_dir = resolve_job_dir(job)
        manifest = load_manifest(job_dir)
        video_of(manifest, link, lambda text: print(text, file=sys.stderr))
        write_tab_score(job_dir, manifest)
        if exports:
            from video2score.run import export_score

            for path in export_score(job_dir, manifest, exports):
                print(path)
        save_manifest(job_dir, manifest)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    summary = manifest["tab"]
    print(
        f"{job_dir / 'score.musicxml'} measures={summary['measures']} notes={summary['notes']} "
        f"measures_with_issues={summary['measures_with_issues']}"
    )
    return 0


def _compare(job: str, reference: str) -> int:
    from pathlib import Path

    from video2score.compare import run_compare
    from video2score.jobs import resolve_job_dir

    try:
        job_dir = resolve_job_dir(job)
        result = run_compare(job_dir, Path(reference))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result.summary())
    print(f"逐小节差异：{job_dir / 'compare.txt'}")
    return 0
