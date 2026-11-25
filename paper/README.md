# 论文工程说明

## 编译状态

本机已安装 MiKTeX 25.12（用户级：`%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`），两版均已实际编译通过。

| 版本 | 页数 | 溢出框 | 未定义引用 |
|---|---:|---:|---:|
| 英文 `en/main.pdf` | 6 | 0 | 0 |
| 中文 `zh/main.pdf` | 7 | 0 | 0 |

重新编译：

```powershell
$env:PATH = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64;$env:PATH"
cd paper\en; pdflatex -interaction=nonstopmode main.tex; bibtex main; pdflatex -interaction=nonstopmode main.tex; pdflatex -interaction=nonstopmode main.tex
```

或直接跑 `paper/build.ps1 -Dir en` / `-Dir zh`。

## 目录

```
paper/
├── framework.md          论文框架（主张、证据链、章节预算）
├── README.md             本文件
├── build.ps1             编译脚本
├── en/                   英文版：main.tex + 5 章 + fig-tradeoff.tex + references.bib + main.pdf
├── zh/                   中文版：同上
├── review/
│   ├── votes.md          逐章评审投票记录
│   └── *-page1.png       首页渲染图（版式核验用）
├── mot-paper-en.zip      含源码与 PDF，可直接上传 Overleaf
└── mot-paper-zh.zip
```

## 已修正的编译问题

1. 中文版正文中的 `**粗体**` 是 Markdown 写法，被原样印成星号 —— 已全部改为 `\textbf{}`。
2. 摘要原本从第一栏折到第二栏，句子断在中间 —— 已改用 `\twocolumn[...]` 让标题与摘要跨栏。
3. 中文版显式指定 `fontset=windows`，避开 ctex 字体探测。

## 投稿前待办

1. **换模板**：现为 `article` 近似双栏，ICDIP 需换官方 `spie.cls`。
2. **双盲**：移除作者信息与仓库链接。
3. **参考文献**：目前 11 条，8 页会议建议扩到 20 条以上，尤其外观重识别方向。
4. **页数**：英文 6 页，离 8 页上限还有余量，可考虑补一张定性示例图或把迁移实验展开。

## 数字口径

全文失败比例均为**观测值**，取自：

- 官方测试主对照 `runs/reconnect_study/personpath22/test-byte.json`、`test-bot.json`
- 接受量对齐 `runs/reconnect_supplement/round2/test-paired-contrasts.csv`
- 消融与外部评分 `runs/reconnect_supplement/round3/paired-effects.csv`、`policy-results.csv`

有限总体风险证书（19.95% 上界）不在本文范围内，见
`../_repo_tmp/mot-reconnection-experiments/docs/submission-scope.md`。
