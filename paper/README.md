# 论文工程说明

## 编译状态

本机已安装 MiKTeX 25.12（用户级：`%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`）。ICDIP 要求正文不少于 8 页、不超过 15 页，两版均满足。

| 版本 | 页数 | 溢出框 | 未定义引用 |
|---|---:|---:|---:|
| 英文 `en/main.pdf` | 8 | 0 | 0 |
| 中文 `zh/main.pdf` | 9 | 0 | 0 |

重新编译：

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64;$env:PATH"
cd paper\en; pdflatex -interaction=nonstopmode main.tex; bibtex main; pdflatex -interaction=nonstopmode main.tex; pdflatex -interaction=nonstopmode main.tex
```

或运行 `paper/build.ps1 -Dir en` / `-Dir zh`。

## 结构

```
main.tex          标题、跨栏摘要、关键字
01-introduction   四段：决策框架、两条现有路线、做法、四条主张
02-related-work   三条线：关联与重连、外观重识别、选择与风险控制
03-method         门控定义 + 特征表（表 1）+ 选点规则
04-experiments    数据、对照、主结果（表 2）、接受量对齐、跨上限扫描（表 3）、
                  门控作用位置、增益来源（表 4）、迁移、代价、范围
05-discussion     代价的语言、交互项为何只在一处有效、外观是补充、开放问题
06-conclusion     收束
```

四张表加一张图：

| 编号 | 内容 |
|---|---|
| 表 1 | 门控的 23 个输入坐标 |
| 表 2 | 官方测试子集全策略结果 |
| 表 3 | 跨 B=6/12/24 的匹配接受量比较（17 个工作点） |
| 表 4 | 交互项消融与 BoostTrack 外部评分配对对照 |
| 图 1 | 失败比例—恢复缺口权衡散点 |

## 为满足 8 页下限所做的扩写

不是灌水，全部基于仓库内已有但未写入论文的数据：

1. **特征表**：`scoring.py` 里 23 维特征的实际定义，此前正文只用省略号带过。
2. **跨上限扫描**：第二轮 `test-paired-contrasts.csv` 中 B=6/12/24 的 17 个锁定工作点，此前只报告了 B=12。
3. **门控作用位置**：第三轮 `diagnostics/*-strata.csv`，按缺口长度分箱的恢复率，以及裕量分箱样本不足的如实说明。
4. **讨论章**：对代价、交互项增益范围、外观关系的解释，无新数字。
5. **相关工作**：补充级联阈值为何是全局的、外观描述子的失效模式、选择与过滤的区别。

## 投稿前待办

1. **换模板**：现为 `article` 近似双栏，需换官方 `spie.cls` 后重新核页数。
2. **双盲**：移除作者信息与仓库链接。
3. **参考文献**：目前 11 条，偏少；外观重识别方向缺引用是最容易被问的。
4. **参考文献与表 3 的一致性**：表 3 中 ByteTrack B=12 低档因校准未对齐而缺项，正文已说明。

## 数字口径

全部为**观测值**，来源：

- 主对照 `runs/reconnect_study/personpath22/test-byte.json`、`test-bot.json`
- 接受量对齐与跨上限 `runs/reconnect_supplement/round2/test-paired-contrasts.csv`
- 消融与外部评分 `runs/reconnect_supplement/round3/paired-effects.csv`、`policy-results.csv`
- 分箱诊断 `runs/reconnect_supplement/round3/diagnostics/`

有限总体风险证书（19.95% 上界）不在本文范围内，见
`../_repo_tmp/mot-reconnection-experiments/docs/submission-scope.md`。
