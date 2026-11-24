# 第二轮近似等接受量实验

[一分钟汇报](round2-brief.zh-CN.md) · [完整交接和实验结论](../runs/reconnect_supplement/round2/round2-handoff.md) · [实验设计](../runs/reconnect_supplement/round2-design.md)

## 查结果

- [配对对照全表](../runs/reconnect_supplement/round2/test-paired-contrasts.csv)保留 34 个配对比较和 1 个校准缺项。测试族间接受量容差与偏离校准目标分别报告。
- [策略指标全表](../runs/reconnect_supplement/round2/test-policy-summary.csv)包含接受、错误、正确重连、自然缺口恢复和整体跟踪指标。
- [ByteTrack 三档](../runs/reconnect_supplement/round2/byte/)与 [BoT-SORT 运动版三档](../runs/reconnect_supplement/round2/bot/)各保留初始及加密阈值配置、校准逐视频结果、策略锁、测试逐视频结果和配对报告。
- [输入核查](../runs/reconnect_supplement/round2/input-audit.json)、[来源资格](../runs/reconnect_supplement/round2/source-eligibility.md)和[独立审计](../runs/reconnect_supplement/round2/independent-audit.json)记录来源划分、文件摘要及复核范围。测试结果属于在已分析过的 74 个视频上做的探索性敏感性分析。

## 重建表格

安装基本依赖后，从仓库根目录运行以下 PowerShell 命令。此步骤直接读取六份已提交的报告。

```powershell
$round = 'runs/reconnect_supplement/round2'
$reports = @(6,12,24 | ForEach-Object { "$round/byte/B$_/report.json" }) + @(6,12,24 | ForEach-Object { "$round/bot/B$_/evaluation.json" })
python -m reconnect_study.supplement_matched_tables @reports --output "$round/rebuilt"
```

## 重跑在线跟踪

按照 [PersonPath22 数据指引](datasets.md#personpath22-主实验)下载标注和发布方的检测结果，并按[主实验复现说明](../reconnect_study/reproduction.md)安装依赖和固定版本的跟踪器。完整校准及测试命令见[第二轮开发记录](../runs/reconnect_supplement/round2/byte/reproduction.md)和[交接记录](../runs/reconnect_supplement/round2/round2-handoff.md)。重新生成校准锁时，建议在独立本地副本运行，以便和已提交的锁逐项比较。

若只重跑锁定测试，保留配置、校准结果和锁的原字节，并指定新的输出文件。以下用 ByteTrack 12 次上限作示例，其他档位只需替换工作目录。BoT-SORT 的配置和锁位于 `bot` 子目录。

```powershell
$manifest = 'runs/reconnect_study/personpath22/manifest.json'
$dir = 'runs/reconnect_supplement/round2/byte/B12'
python -m reconnect_study.supplement_matched_eval "$dir/traces/rerun-test.json" "$dir/locked.json" "$dir/traces/rerun-report.json" --run-manifest $manifest --run-config "$dir/config.json" --trace-directory "$dir/traces/rerun"
```

本仓库提供逐视频统计和轨迹哈希。公开数据、检测缓存和完整逐帧轨迹下载或重跑后保存在本地。若要使用审计脚本检查逐帧轨迹，先按交接记录在本地生成对应目录。
