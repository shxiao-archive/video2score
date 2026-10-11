# 视频动态乐谱转谱流水线（精简架构速查）

> 完整设计说明与架构细节参见：[SPECIFICATION.md](SPECIFICATION.md)

---

## 1. 核心流水线

```text
视频源 (默认每 0.33 秒取一帧)
  │  frames.score_box      --> 找谱面区域（整页白纸，或横贯画面的谱带）
  │  staff.find_systems    --> 找谱线（4/5/6条等距长横线；五线谱与 TAB 配对；找小节线）
  │  measures              --> 切出两侧都有小节线的完整小节；提取小节号 ROI
  ▼
track.Tracker              --> 跨帧位移追踪：算整体位移，小节对应到已有槽位
  │  ocr / numbers         --> 局部对比度二值化、投影切分、灰度模板匹配读号
  │  order.arrange         --> 拆开误并、按最长递增子序列排序、补漏号、清离群号
  │  clean.render_slot     --> 帧间对齐、去彩色光标、中位数多帧投票、重画谱线
  ▼
layout / header            --> 统一线距，行宽微调补齐，输出 score.pdf
  │  tab / rhythm / marks  --> 读品位数字、符干/符杠/连音时值、演奏法特征
  │  tabscore              --> 拍号守恒修正，组装双行谱 → score.musicxml
  ▼
gpfile / engrave           --> 导出 .gp / .gpx / .gp5，通过 alphaTab 输出印刷级 PDF
```

---

## 2. 核心记号判定速查

| 记号 | 判定规则（线距 $S_p$ 口径） | 输出到 MusicXML |
|---|---|---|
| **拍号** | 小节开头上下叠着的两个粗体数字，分母只在 2/4/8/16 中挑 | `<time>` |
| **调号** | 紧跟谱号后：升号为双竖笔（间距 $<0.6 S_p$），降号为单竖笔加圆肚 | `<key>` |
| **调弦** | 行首前缀图每条弦高度最右侧字母，在标准调弦 $\pm 2$ 半音内挑 | `<staff-tuning>` |
| **速度** | 第一小节上方 "♩ = 数字"：先找等号，再读右侧数字 | `<sound tempo>` |
| **品位** | 弦线附近找字；抹去符干与小节线，被弦线切断的笔画拼回 | `<string>` / `<fret>` |
| **括号延音** | TAB 上带括号的数字，代表延音 | 延音线，不新起音 |
| **闷音 (Mute)** | TAB 上为 X；五线谱上 X 符头中心交叉，无实心/空心核 | `<notehead>x</notehead>` |
| **滑音 (Slide)** | 倾斜线段：按高低端与符头高度差分滑入/滑出；带 "sl." 或弧线为连奏 | `<slide>` / `<slur>` |
| **击弦/勾弦** | 跨音符弧线（中粗两端尖）：后音品位升高为击弦，降低为勾弦 | `<hammer-on>` / `<pull-off>` |
| **装饰音** | TAB 上小一号的数字（高宽比 $\ge 1.3$） | `<grace slash="yes"/>` |
| **跳音** | 符头另一侧 $0.5 \sim 2 S_p$ 处的孤立实心小圆点 | `<staccato/>` |
| **重音** | 符头另一侧 $1.2 \sim 4.5 S_p$ 处的尖朝外 `^` 或尖朝右 `>` | `<strong-accent/>` / `<accent/>` |
| **拍弦/勾弦** | TAB 上方正对音符的 "S"（拍）或 "P"（勾）字母 | `<?GP <root><bass-attack...>>` |
| **推弦** | TAB 上方垂直实心箭头，上方小数字标明推音幅度 | `<bend-alter>` |
| **符杠/符尾** | 符干端部相连的横向粗墨迹段（约 $0.45 S_p$ 厚）；整列长竖线为光标，剔除 | 符杠分组；单出小钩写 `forward hook` |

---

## 3. 关键异常处理原则

- **小节号冲突**：两边都清楚读出号但号码不同时，坚决不配对，放弃该位移。
- **光标切开音符**：上一帧对齐失败时，向前检索最近几帧同一位置长相完好的小节接上。
- **加线误判**：低音区密集加线比正常谱线稀疏、单段短。存疑的 6 线组降低优先级，选错开一线的 5 线组。
- **五弦 TAB 消歧**：上下两组线距相差 >15% 时，线距大的是 TAB；仅一组 5 线无法区分时用 `--notation tab`。
- **清洗图补漏与仲裁**：各帧投票后再读一遍清洗图。投票里没有的音，清洗图读得有把握就补进来；投票读得没把握、或只读出两位数里的一位时，清洗图在同一处读得很有把握（$\ge 0.85$）才换成清洗图的读数；其余只补不改。
- **拍号守恒修正**：小节时值超限时，依序尝试去掉误读休止符、连杠音四分改八分、单一三连音或附点修正。

---

## 4. 文件与产物结构

```text
jobs/<id>/
├── manifest.json              # 作业元数据与制品索引
├── measures.json              # 小节库：几何坐标、谱线参数、序号置信度
├── measures/
│   ├── 0001.png               # 清洗重绘后的小节图
│   └── 0001_raw.png           # 最佳参考原图（另有 raw1... 等辅助投票帧）
├── score.pdf                  # 视频截图清洗拼接 PDF
├── score.musicxml             # 识别出的五线谱 + TAB
├── score.gp / .gpx / .gp5     # 导出的 Guitar Pro 文件
└── tab_issues.json / .txt     # 需人工核对的可疑小节（界面用 json，txt 给人看）
```

---

## 5. 常用命令

```powershell
# 1. 基础转换
python -m video2score run --source "<视频链接/BV号/本地文件>" [--export pdf,gp,gp5]

# 2. 启动本地 Web 界面
python -m video2score app [--port 17865] [--no-browser]

# 3. 对已有作业重新排版 PDF（无需重新下载视频）
python -m video2score pdf --job <job_id> [--link "<原视频链接>"]

# 4. 对已有作业单独重跑 TAB 识别
python -m video2score tab --job <job_id> [--export gp,pdf]

# 5. 多格式导出
python -m video2score export --job <job_id> --to gp,gpx,gp5,pdf [--out <输出目录>]

# 6. 回归与基准测试（测试和素材只在开发者本地）
python -m unittest discover -s tests
python -m video2score.bench
python -m video2score bench-pdf [--rerun]
```
