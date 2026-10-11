# vendor

这里是浅克隆的上游仓库，只在本机阅读或运行。本仓库不修改这些目录，也不把它们的源码抄进自己的代码。

`vendor/*/` 已写入根目录 `.gitignore`。换机器时按下面的地址重新克隆，不要把这三个目录提交进来。

记录日期：2026-09-27。均为 `git clone --depth 1`。

2026-09-28 起，`video2score` 运行时不再导入这里的任何代码。`bili-tab-capture` 和 `score_capture` 只作对照；`oemer` 留给下一轮的五线谱识别。没有克隆这三个目录，`python -m video2score run` 也能跑。

| 目录 | 地址 | 分支 | Commit | 许可证 | 在本框架里的位置 |
| --- | --- | --- | --- | --- | --- |
| `bili-tab-capture` | https://gitee.com/m1ku666/bili-tab-capture.git | `main` | `ea6d3e55efd41c799d7c8d6e3b493de54a7417c8`（v3.6，2026-09-25） | 仓库未声明 | 只读对照（2026-09-28 前是阶段一的运行时依赖） |
| `score_capture` | https://github.com/Carrot-shreds/score_capture.git | `master` | `94951b8cdc4c2059ad54c2cde63ef647d91a129f`（2026-02-27） | GPL-3.0 | 只读对照。小节线与裁切思路的来源，不导入、不复制 |
| `oemer` | https://github.com/BreezeWhite/oemer.git | `main` | `dbe2a933d630d0f74805d717960eb259473f5978`（2025-04-18） | MIT | 五线谱识别备查。这一步不接入；权重在首次运行时另下，不提交进 git |

不放进 `vendor/` 的：

- [santiRostan/TabCapture](https://github.com/santiRostan/TabCapture)：BiliTabCapture 的上游，YouTube 下载已被拿掉，本框架不再单独使用。
- [Audiveris](https://github.com/Audiveris/audiveris)：AGPL-3.0。五线谱识别用本机已安装的独立程序，不克隆进本仓库。
- Guitar Pro、Soundslice：商业或网页服务，不是本仓库的依赖。

阶段一在本机运行（不要改 `vendor/bili-tab-capture` 里的文件）：

```powershell
cd vendor\bili-tab-capture
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
$env:BTAB_NO_BROWSER = "1"   # 不想自动打开浏览器时加上
.\.venv\Scripts\python app.py
```

默认地址是 `http://127.0.0.1:5000`。端口被占用时会往后顺延。
