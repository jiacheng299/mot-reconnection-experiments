# 论文工程说明

## 编译状态

MiKTeX 25.12（用户级：`%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`）。ICDIP 要求 8–15 页。

| 版本 | 页数 | 溢出框 | 未定义引用 | 参考文献（库/实际引用） |
|---|---:|---:|---:|---:|
| 英文 `en/main.pdf` | 10 | 0 | 0 | 22 / 20 |
| 中文 `zh/main.pdf` | 11 | 0 | 0 | 22 / 20 |

重新编译：

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64;$env:PATH"
cd paper\en; pdflatex -interaction=nonstopmode main.tex; bibtex main; pdflatex -interaction=nonstopmode main.tex; pdflatex -interaction=nonstopmode main.tex
```

或运行 `paper/build.ps1 -Dir en` / `-Dir zh`。

## 结构

```
main.tex            标题、跨栏摘要、关键字
01-introduction     决策框架、两种部署情境、已有两条路线、四条主张、边界
02-related-work     关联与重连 · 外观重识别 · 选择与风险控制 · 基准与计分
03-method           门控定义、特征表（表 1）、地位示意（图 1）、
                    坐标分组解释、选点规则
04-experiments      数据 · 对照 · 主结果（表 2）· 接受量对齐 ·
                    跨上限扫描（表 3）· 门控作用位置 · 增益来源（表 4）·
                    定性示例 · 实现与复现 · 有效性威胁
05-discussion       主张—证据对照（表 5）、代价、交互项范围、外观关系、开放问题
06-conclusion       收束
```

五张表、两张图。

## 三轮扩写的来源

全部基于仓库内已有数据，没有新跑实验：

| 轮次 | 增补 | 出处 |
|---|---|---|
| 第一轮（6→8 页） | 特征表、跨上限扫描、分箱诊断、讨论章、相关工作三段 | `scoring.py`、round2/round3 CSV |
| 第二轮（8→9 页） | 基准与计分子节、定性示例、实现与复现、有效性威胁 | `experiment_record.md`、配置记录 |
| 第三轮（9→10 页） | 门控示意图（TikZ）、坐标分组解释、部署情境段、主张—证据表 | 无新数据，叙述性 |

## 投稿前待办

1. **换模板**：现为 `article` 近似双栏，需换官方 `spie.cls` 后重新核页数。
2. **双盲**：移除作者信息与仓库链接。
3. **参考文献 13 条**：仍偏少，外观重识别方向只有 BoT-SORT 与 FairMOT 两条，
   Tracktor / CenterTrack 因未查到可靠条目暂缺。
4. **表 3 缺项**：ByteTrack B=12 低档因校准未对齐，正文已说明。

## 数字口径

全部为**观测值**：

- 主对照 `runs/reconnect_study/personpath22/test-byte.json`、`test-bot.json`
- 接受量对齐与跨上限 `runs/reconnect_supplement/round2/test-paired-contrasts.csv`
- 消融与外部评分 `runs/reconnect_supplement/round3/paired-effects.csv`、`policy-results.csv`
- 分箱诊断 `runs/reconnect_supplement/round3/diagnostics/`

有限总体风险证书（19.95% 上界）不在本文范围内，见
`../_repo_tmp/mot-reconnection-experiments/docs/submission-scope.md`。
