# 智能体接手指南

1. 阅读根目录 README 和 reconnect_study/experiment_record.md，先确认是在继续身份重连主线，还是查看旧 CGHTrack 实验。
2. 从 runs/reconnect_study/personpath22/manifest.json 查视频划分、数据来源、逐序列摘要。PersonPath22 由数据集提供人工框和身份，FCOS 提供候选框与分数，跟踪器输出临时编号。人工标注只用于评分。
3. 开发集用于训练与网格固定，校准集用于选运行点，官方测试子集用于确认性比较。随机有限总体风险分析是另一个划分，有视频重叠。保持这两份证据的含义。
4. test-byte.json、test-bot.json 有逐视频与合计结果；effects-*.json 有按视频配对重采样；finite-certificate-result.json 有两个风险指标。完整追溯路径见 docs/artifacts.md。
5. 改动模块后运行 python -m unittest discover -s reconnect_study/tests -v。先用开发划分验证，再决定是否生成新的独立评测。保留冻结配置和清单的原文本字节，因为它们的哈希用在风险验证里。
6. 模块接入在 reconnect_study/policy.py 与 tracker.py，候选标签在 events.py，统计证明和实现分别在 statistical_method.md 与 certification.py。
7. 旧项目按 docs/legacy-workspace-agents.md 与 P0–P6 记录查找。记录中的桌面文件和历史大体积原始产物属于本机环境。

新实验记录说明输入、处理、输出、实测结果、适用范围和验收办法。写作时分开明确标注自动检测、人工真值、跟踪器输出，以及实测失败率和统计上界。
