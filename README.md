# MOT 在线身份重连实验

这是论文实验交接仓库。主线研究多目标跟踪器（MOT）执行一次旧身份重连时，如何减少接错或无法核验的动作，并量化因此损失的轨迹恢复。此前 CGHTrack P0–P6 的实验记录和关键结果也保存在仓库中，供追溯。

## 从这里开始

| 需要了解 | 入口 |
|---|---|
| 给接手智能体的操作约定 | [AGENTS.md](AGENTS.md) |
| 任务、方法、真实结果和适用范围 | [中文实验记录](reconnect_study/experiment_record.md) |
| 英文论文稿 | [方法](reconnect_study/method.tex) · [实验](reconnect_study/experiments.tex) · [PDF](runs/reconnect_study/publication/publication_preview.pdf) |
| 第二轮补实验 | [一分钟汇报](docs/round2-brief.zh-CN.md) · [数据与复现入口](docs/round2.md) |
| 获取数据与准备本机环境 | [数据下载指引](docs/datasets.md) · [复跑步骤](reconnect_study/reproduction.md) |
| 查原始统计结果 | [ByteTrack 测试](runs/reconnect_study/personpath22/test-byte.json) · [BoT-SORT 测试](runs/reconnect_study/personpath22/test-bot.json) · [归档风险证据（未进入投稿版本）](runs/reconnect_study/personpath22/finite-certificate-result.json) · [复现审计](runs/reconnect_study/reproducibility-audit.json) |
| 投稿版本的取材范围 | [研究范围说明](docs/submission-scope.md) |

PersonPath22 的官方测试子集含 74 个视频和 3,155 个自然检测缺口。在相同重连限额下，ByteTrack 的可评估失败比例由 46.77% 降到 35.39%，正确执行的重连由 239 增至 283，恢复自然缺口数由 388 增至 393。不限额原版恢复 529 个缺口。评分训练使用本机 RTX 3060 Ti。投稿版本只报告观测到的失败比例，不包含有限总体风险证书，取舍理由见研究范围说明。

## 运行入口

安装 Python 3.13 和项目依赖。复现 CUDA 训练使用 PyTorch 2.7.1 + CUDA 12.6。仓库记录了原环境包的版本，安装前可根据系统及硬件调整二进制版本。官方依赖下载与检查点信息见数据指引。

```powershell
python tools/bootstrap_vendors.py
python -m unittest discover -s reconnect_study/tests -v
```

仓库提供实验代码、模型评分配置、结果摘要与逐视频统计；公开数据仅提供来源、划分和摘要。较大的检测缓存、逐帧轨迹、原始媒体、虚拟环境及临时文件留在原工作区。

## 第三轮补实验

[一页验收说明](runs/reconnect_supplement/round3/round3-handoff.zh-CN.md) · [策略结果](runs/reconnect_supplement/round3/policy-results.csv) · [配对差值](runs/reconnect_supplement/round3/paired-effects.csv) · [主图](runs/reconnect_supplement/round3/main-effects.pdf)

用归档逐视频统计重建表格和主图。此步骤使用仓库内文件。

```powershell
python -m reconnect_study.round3_summary
python -m reconnect_study.round3_figure
```

完整闭环复跑需按[数据指引](docs/datasets.md)准备冻结输入，并按[消融复现步骤](runs/reconnect_supplement/round3/ablation/reproduction.md)和 [BoostTrack 评分复现步骤](runs/reconnect_supplement/round3/boost/reproduction.md)在独立工作副本运行。开发数值核验使用以下固定的作者代码。

```powershell
git clone https://github.com/vukasin-stanojevic/BoostTrack.git runs/reconnect_supplement/round3/boost/upstream/BoostTrack
git -C runs/reconnect_supplement/round3/boost/upstream/BoostTrack checkout --detach fb5bfc3a8f067476565e753b3a73df4d757c9d03
```

完整回放在本地重建逐帧轨迹。HOTA 独立核验读取这些轨迹，配对重采样读取逐视频统计。ByteTrack 使用宿主状态坐标重排，运动版 BoT-SORT 使用一阶协方差近似。第三轮复用此前分析过的 74 个官方测试视频，属于探索性敏感性分析。
