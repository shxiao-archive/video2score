<p align="center"><img src="packaging/icons/video2score.png" width="96" alt="video2score Logo"></p>

<h1 align="center">video2score 视频转谱</h1>

<p align="center">将贝斯、吉他动态乐谱视频全自动转换为高清乐谱 PDF 与 Guitar Pro / MusicXML 文件。</p>

<p align="center">
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/shxiao-archive/video2score/releases/latest"><img src="https://img.shields.io/github/v/release/shxiao-archive/video2score" alt="Latest Release"></a>
</p>

---

## 核心特性

- **去噪拼接 PDF**：跨帧追踪小节，逐像素中位数投票过滤播放光标、变色高亮与动态背景，输出整洁统一的 A4 乐谱。
- **符号与技巧识别**：识别 TAB 品位、五线谱时值，以及击勾弦、推弦、滑音、闷音、拍弦等演奏法。
- **多格式导出**：支持输出截图重排 PDF、矢量排版 PDF、MusicXML 以及 Guitar Pro（`.gp` / `.gpx` / `.gp5`）文件。
- **开箱即用**：提供免安装桌面包，内置轻量本地 Web 界面，无需配置 Python 或 Node.js 环境。

---

## 快速开始

### 方式一：下载免安装包 (推荐)

前往 **[Releases](https://github.com/shxiao-archive/video2score/releases/latest)** 下载对应系统的压缩包，解压后直接运行：

| 系统平台 | 架构 | 文件名 |
|---|---|---|
| **Windows** | x64 | `video2score-<版本>-windows-x64.zip` |
| **macOS** | Apple Silicon (M 系列芯片) | `video2score-<版本>-macos-arm64.zip` |
| **macOS** | Intel x64 | `video2score-<版本>-macos-x64.zip` |

**使用步骤**：
1. **启动程序**：Windows 双击 `video2score.exe`，macOS 双击 `video2score.app`，浏览器将自动打开操作页面。
2. **提交任务**：粘贴视频链接（B 站链接 / BV 号 / YouTube 链接）或拖入本地视频文件，选择导出格式后点击“开始转谱”。
3. **获取产物**：转换完成后，在“我的谱”中查看和下载对应文件。对低置信度的小节，可在“看要核对的小节”中结合原图核对。

---

### 方式二：从源码运行 (开发者)

#### 环境要求
- Python 3.13（开发和打包都用这个版本）
- Node.js 22.12 以上（导出排版 PDF 用）
- 推荐使用 Windows 或 macOS

#### 安装与启动
```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
# 导出排版 PDF 还要装 alphaTab、Puppeteer，并下载打印用的 chrome-headless-shell（只需一次）
cd video2score/alphatab; npm install; cd ../..
python packaging/build.py --browser

# 3. 运行本地 Web 服务
python -m video2score app

# 或直接通过命令行处理视频
python -m video2score run --source "<视频链接或文件路径>" --export pdf,gp
```

---

## 常见问题 (FAQ)

- **macOS 提示“无法验证开发者”？**  
  初次打开时，在访达（Finder）中右键点击 `video2score.app`，选择“打开”，在弹出窗口中再次点击“打开”即可。
- **YouTube 视频下载较慢或失败？**  
  请确认当前网络环境可正常访问 YouTube。若使用代理工具，请开启系统全局代理后再试。
- **B 站视频清晰度不足影响识别？**  
  点击界面右上角“B 站登录”，使用手机 App 扫码登录。登录后将下载更高清的视频流，提升识别准确率。
- **转谱生成的数据保存在哪里？**  
  数据均保存在程序根目录下的 `data/` 目录（`jobs/`、`uploads/`、`trash/` 等）。删除整个 `video2score` 文件夹即可完全卸载。

---

## 开发者文档

流水线的算法细节、模块设计、数据契约与测试基准参见：
- **[完整设计规格说明书](docs/SPECIFICATION.md)**：包含时空追踪、OCR 模板匹配、乐理判据矩阵、数据契约与合规说明。
- **[精简架构速查](docs/ARCHITECTURE.md)**：核心数据流拓扑、四大架构不变量、记号判定规则表与 CLI 手册。

---

## 版权与免责声明

- 本工具转录输出的乐谱仅供个人学习、研究与练习使用。
- 生成谱面页脚默认标注原视频作者与出处链接，请尊重制谱者与版权方的合法权益，严禁商用或用于盗版传播。
- 若原作者或官方提供了正版乐谱购买渠道，请优先购买正版支持原作者。

---

## 开源许可证

本项目源码采用 [MIT License](LICENSE.txt) 开源。

发布的独立运行包中包含部分第三方组件（例如 PyMuPDF 为 AGPL-3.0 许可），各组件许可证全文及来源说明详见发行包中的 `THIRD_PARTY_NOTICES.txt`。
