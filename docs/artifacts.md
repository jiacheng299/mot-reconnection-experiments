# 结果文件索引

| 内容 | 目录或文件 |
|---|---|
| 当前论文实验正文、数学与复跑说明 | reconnect_study/ |
| 冻结配置、开发/校准/官方测试、有限总体风险、逐视频计算 | runs/reconnect_study/personpath22/ |
| BDD100K 探索和 DanceTrack 迁移 | runs/reconnect_study/bdd100k/、runs/reconnect_study/dance_coco/ |
| 投稿预览、正式图与源点 | runs/reconnect_study/publication/ |
| 数据、哈希、环境和结果的独立复现证据 | runs/reconnect_study/reproducibility-audit.json |
| 旧方案执行记录、验收汇总和恢复试验 | 根目录 P0–P6 与追踪器改进 Markdown、runs/p6-acceptance/ 等精选输出 |
| 第二轮近似等接受量实验 | [第二轮复现入口](round2.md) · [逐视频数据、阈值锁与报告](../runs/reconnect_supplement/round2/) |
| 不包含在仓库的本地大文件 | 数据集媒体和本地标注副本、检测缓存、全量原始轨迹、下载暂存、虚拟环境 |

正文字段保留原测试复现所需的设置。主线评分模型的数值权重包含在 byte-config.json 与 bot-config.json 内。
