"""打包 video2score 免安装包：python packaging/build.py → dist/video2score-<版本>-<windows|macos>-<x64|arm64>.zip

在要打包的那个平台上跑（PyInstaller 不能跨平台打包）；CI 见 .github/workflows/release.yml。需要：
- 装好 requirements.txt 和 requirements-build.txt 的 Python；
- npm（装 alphaTab、Puppeteer 用；包里带的 Node 另外下载）。

步骤：下载固定版本的 Node、chrome-headless-shell 并核对 sha256 → npm ci 装 alphaTab 和 Puppeteer → PyInstaller 打包 →
放进说明、许可证、第三方声明 → 跑一次自检（--selftest）→ 打成 zip。中间文件都在 build/，结果在 dist/。

python packaging/build.py --browser 只下载 chrome-headless-shell 到 video2score/alphatab/chrome/（开发时导出排版 PDF 用，和打包的同一版本）。
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from video2score import __version__  # noqa: E402

NODE_VERSION = "v24.21.0"  # Node LTS
# 打印 PDF 用的 chrome-headless-shell（Google 的 Chrome for Testing）。Google 不发布 sha256，下面是核对过
# 下载服务器给的 md5 以后自己算的；换版本时三个都要重算。
BROWSER_VERSION = "155.0.8059.39"
BROWSER_SHA256 = {
    "win64": "20798f7c51a22def0b3d02e526a8f52e7ee2c7eddcde9d81a4c23f618e8b5436",
    "mac-arm64": "b3e093c06001c41e68decbc8dd4a62f9efe4bf4e4dd247a686ea531863448d75",
    "mac-x64": "6338a784c691f42dd1ed6aeeef650171c7d72cf74bdb8450d0079864e72a8074",
}
BUILD = ROOT / "build"
DIST = ROOT / "dist"
APP = "video2score"


def main() -> int:
    system, arch = _platform()
    if "--browser" in sys.argv[1:]:
        target = _browser(system, arch, ROOT / "video2score" / "alphatab" / "chrome")
        print(f"已下载：{target}", flush=True)
        return 0
    print(f"打包 {APP} {__version__}（{system}-{arch}）", flush=True)
    for name in ("glyphs.npz", "font_widths.json"):
        if not (ROOT / "video2score" / "resources" / name).is_file():
            raise SystemExit(f"缺少 video2score/resources/{name}：先在 Windows 上运行 python -m video2score glyphs")
    _node(system, arch)
    _browser(system, arch, BUILD / "chrome")
    _alphatab()
    _pyinstaller()
    release = _assemble(system)
    _selftest(release, system)
    archive = _zip(release, f"{APP}-{__version__}-{system}-{arch}", system)
    print(f"完成：{archive}", flush=True)
    return 0


def _platform() -> tuple[str, str]:
    system = {"win32": "windows", "darwin": "macos"}.get(sys.platform)
    if system is None:
        raise SystemExit("只支持在 Windows 或 macOS 上打包。")
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    return system, arch


# ---------------------------------------------------------------- Node


def _node(system: str, arch: str) -> Path:
    """下载 Node 官方发布的压缩包（按官方 SHASUMS256.txt 核对），取出 node 可执行文件放到 build/node/。"""
    target = BUILD / "node" / ("node.exe" if system == "windows" else "node")
    stamp = target.with_suffix(".version")
    if target.is_file() and stamp.is_file() and stamp.read_text().strip() == f"{NODE_VERSION}-{arch}":
        return target
    platform_name = "win" if system == "windows" else "darwin"
    name = f"node-{NODE_VERSION}-{platform_name}-{arch}" + (".zip" if system == "windows" else ".tar.gz")
    base = f"https://nodejs.org/dist/{NODE_VERSION}/"
    print(f"下载 {name}", flush=True)
    sums = _download(base + "SHASUMS256.txt").decode("ascii")
    expected = next((line.split()[0] for line in sums.splitlines() if line.endswith("  " + name)), None)
    if expected is None:
        raise SystemExit(f"SHASUMS256.txt 里没有 {name}")
    data = _download(base + name)
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f"{name} 的 sha256 不对")
    folder = name.rsplit(".zip", 1)[0].rsplit(".tar.gz", 1)[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    if system == "windows":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            target.write_bytes(archive.read(f"{folder}/node.exe"))
            (target.parent / "LICENSE").write_bytes(archive.read(f"{folder}/LICENSE"))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            target.write_bytes(archive.extractfile(f"{folder}/bin/node").read())
            (target.parent / "LICENSE").write_bytes(archive.extractfile(f"{folder}/LICENSE").read())
        target.chmod(0o755)
    stamp.write_text(f"{NODE_VERSION}-{arch}\n")
    return target


def _browser(system: str, arch: str, folder: Path) -> Path:
    """下载固定版本的 chrome-headless-shell，核对 sha256，解压到 folder（去掉压缩包里最外层的文件夹）。"""
    platform_name = "win64" if system == "windows" else f"mac-{arch}"
    executable = folder / ("chrome-headless-shell.exe" if system == "windows" else "chrome-headless-shell")
    stamp = folder / "VERSION.txt"
    if executable.is_file() and stamp.is_file() and stamp.read_text().strip() == f"{BROWSER_VERSION}-{platform_name}":
        return executable
    name = f"chrome-headless-shell-{platform_name}"
    url = f"https://storage.googleapis.com/chrome-for-testing-public/{BROWSER_VERSION}/{platform_name}/{name}.zip"
    print(f"下载 {name} {BROWSER_VERSION}", flush=True)
    data = _download(url)
    if hashlib.sha256(data).hexdigest() != BROWSER_SHA256[platform_name]:
        raise SystemExit(f"{name}.zip 的 sha256 不对")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            relative = member.filename.split("/", 1)[1] if "/" in member.filename else ""
            if not relative or member.is_dir():
                continue
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(member))
            # zipfile 不还原权限：mac 上的可执行文件、动态库按压缩包里记的权限设回去。
            mode = member.external_attr >> 16
            if mode & 0o777:
                path.chmod(mode & 0o777)
    stamp.write_text(f"{BROWSER_VERSION}-{platform_name}\n")
    return executable


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


# ---------------------------------------------------------------- alphaTab、PyInstaller


def _alphatab() -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("没找到 npm：装 alphaTab、Puppeteer 要用（装 Node.js 就有）。")
    print("npm ci（alphaTab、Puppeteer）", flush=True)
    subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=ROOT / "video2score" / "alphatab", check=True)


def _pyinstaller() -> None:
    print("PyInstaller", flush=True)
    subprocess.run(
        [
            sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "video2score.spec"),
            "--noconfirm", "--clean", "--distpath", str(BUILD / "dist"), "--workpath", str(BUILD / "work"),
        ],
        cwd=ROOT,
        check=True,
    )


# ---------------------------------------------------------------- 组装


def _assemble(system: str) -> Path:
    """build/release/video2score/：程序（exe + _internal，或 .app）、README.md、LICENSE.txt、THIRD_PARTY_NOTICES.txt。"""
    release = BUILD / "release" / APP
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)
    if system == "windows":
        shutil.copytree(BUILD / "dist" / APP, release, dirs_exist_ok=True)
    else:
        # .app 里有符号链接，原样拷贝。
        app = release / f"{APP}.app"
        shutil.copytree(BUILD / "dist" / f"{APP}.app", app, symlinks=True)
        # 浏览器原样放进 Resources（不经过 PyInstaller，Google 的签名不动），再给整个 .app 重新做本地签名，
        # 不然 .app 的资源封印对不上，mac 会说"已损坏"。
        shutil.copytree(BUILD / "chrome", app / "Contents" / "Resources" / "chrome", symlinks=True)
        subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True)
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)
    shutil.copy2(ROOT / "packaging" / "README.md", release / "README.md")
    license_file = ROOT / "LICENSE.txt"
    if license_file.is_file():
        shutil.copy2(license_file, release / "LICENSE.txt")
    else:
        print("提醒：仓库根目录没有 LICENSE.txt，包里不放本项目的许可证。", flush=True)
    (release / "THIRD_PARTY_NOTICES.txt").write_text(_notices(), encoding="utf-8")
    return release


def _notices() -> str:
    """第三方声明：包里带的 Python 库（按装好的发行包元数据，连同依赖）、Node.js、alphaTab 和它的 Bravura 字体，附上各自的许可证全文。"""
    parts = [
        f"{APP} {__version__} 带了下面这些第三方软件。各自的版权和许可证如下。\n"
        f"{APP} {__version__} includes the third-party software listed below, under their own licenses.\n"
    ]
    seen: set[str] = set()
    queue = [line.split("==")[0].split(">=")[0].strip() for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    queue += ["certifi", "cffi"]
    while queue:
        name = queue.pop(0)
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for requirement in dist.requires or []:
            if "extra ==" not in requirement:
                queue.append(requirement.split(";")[0].split("[")[0].split("<")[0].split(">")[0].split("=")[0].split("!")[0].split("~")[0].strip())
        meta = dist.metadata
        license_name = meta.get("License-Expression") or meta.get("License") or ""
        if len(license_name) > 80 or not license_name:
            license_name = next((value.split("::")[-1].strip() for value in meta.get_all("Classifier") or [] if value.startswith("License ::")), "见下方许可证全文")
        texts = [
            file for file in dist.files or []
            if any(word in file.name.upper() for word in ("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS")) and file.suffix.lower() in ("", ".txt", ".md", ".rst")
        ]
        body = "\n\n".join(_read(dist.locate_file(file)) for file in texts)
        parts.append(_section(f"{meta['Name']} {dist.version}", license_name, meta.get("Home-page") or "", body))
    node_license = BUILD / "node" / "LICENSE"
    parts.append(_section(f"Node.js {NODE_VERSION}", "MIT（及其附带组件的许可证）", "https://nodejs.org/", _read(node_license)))
    parts.append(
        _section(
            f"chrome-headless-shell {BROWSER_VERSION}（Chromium，Chrome for Testing）",
            "BSD-3-Clause（及其附带组件的许可证，见下方全文）",
            "https://developer.chrome.com/blog/chrome-for-testing",
            _read(BUILD / "chrome" / "LICENSE.headless_shell"),
        )
    )
    # npm 装的包：alphaTab（MPL-2.0）、Puppeteer（Apache-2.0）和它们的依赖，按各自的 package.json 和许可证文件。
    modules = ROOT / "video2score" / "alphatab" / "node_modules"
    for manifest in sorted(modules.glob("*/package.json")) + sorted(modules.glob("@*/*/package.json")):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        texts = [path for path in sorted(manifest.parent.iterdir()) if path.is_file() and path.name.upper().startswith(("LICENSE", "LICENCE", "NOTICE", "COPYING"))]
        homepage = info.get("homepage") or ""
        body = "\n\n".join(_read(path) for path in texts)
        parts.append(_section(f"{info['name']} {info.get('version', '')}", str(info.get("license") or "见下方许可证全文"), homepage, body))
    parts.append(_section("Bravura 音乐字体（随 alphaTab 附带）", "SIL Open Font License 1.1", "https://github.com/steinbergmedia/bravura", ""))
    return "\n".join(parts)


def _section(title: str, license_name: str, url: str, body: str) -> str:
    rule = "=" * 78
    return f"\n{rule}\n{title}\n许可证 / License: {license_name}\n{url}\n{rule}\n\n{body.strip()}\n"


def _read(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------- 自检、打包


def _selftest(release: Path, system: str) -> None:
    executable = release / (f"{APP}.exe" if system == "windows" else f"{APP}.app/Contents/MacOS/{APP}")
    print("自检", flush=True)
    subprocess.run([str(executable), "--selftest"], check=True, timeout=600)
    # 自检在包旁边建的 data/（日志）不放进发布包。
    shutil.rmtree(release / "data", ignore_errors=True)


def _zip(release: Path, name: str, system: str) -> Path:
    DIST.mkdir(exist_ok=True)
    archive = DIST / f"{name}.zip"
    archive.unlink(missing_ok=True)
    if system == "macos":
        # ditto 保留 .app 里的符号链接和可执行权限（Python 的 zipfile 会丢）。
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(release), str(archive)], check=True)
    else:
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(release.rglob("*")):
                handle.write(path, Path(APP) / path.relative_to(release))
    return archive


if __name__ == "__main__":
    raise SystemExit(main())
