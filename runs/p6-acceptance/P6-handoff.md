# P6 · E1 / E7 / E8 结果核验

分数由完整 TrackEval 聚合得到。CGHTrack 报告三个种子的均值 ± 样本标准差。HOTA 综合衡量框检测和身份关联，DetA 衡量检测准确度，AssA 衡量关联准确度。IDF1 衡量预测身份与人工身份匹配的精确率和召回率，MOTA 汇总误检、漏检和身份切换。TrackEval 将这五项按 100 倍标度显示，越高越好，MOTA 可为负。IDSW、CLR_FP、CLR_FN 分别是身份切换、误检、漏检的次数，越低越好。

## MOT17 local held-out · 两条视频

| 方法 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | CLR_FP | CLR_FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack | 45.592 | 41.158 | 51.173 | 53.986 | 40.991 | 156.000 | 3040.000 | 9242.000 |
| OC-SORT | 43.089 | 37.774 | 49.455 | 50.795 | 42.613 | 111.000 | 857.000 | 11128.000 |
| BoT-SORT | 47.939 | 41.779 | 55.443 | 58.066 | 44.966 | 101.000 | 2012.000 | 9487.000 |
| TrackTrack | 45.954 | 40.940 | 52.122 | 54.394 | 40.232 | 81.000 | 3221.000 | 9296.000 |
| CGHTrack | 38.108 ± 2.711 | 35.711 ± 1.427 | 41.133 ± 4.063 | 43.261 ± 3.931 | 33.793 ± 3.541 | 369.333 ± 223.505 | 2536.000 ± 333.636 | 11049.667 ± 425.602 |
| CAMELTrack | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 |
| HyperMOT | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 |

## MOT20 train-transfer · 四条视频

| 方法 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | CLR_FP | CLR_FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack | 41.340 | 51.779 | 33.448 | 52.970 | 66.908 | 13468.000 | 159062.000 | 202939.000 |
| OC-SORT | 40.321 | 49.251 | 33.409 | 53.336 | 67.083 | 11009.000 | 39517.000 | 322958.000 |
| BoT-SORT | 44.131 | 52.801 | 37.315 | 58.546 | 70.666 | 10278.000 | 100741.000 | 221813.000 |
| TrackTrack | 40.635 | 50.863 | 32.910 | 54.105 | 65.152 | 13191.000 | 189070.000 | 193129.000 |
| CGHTrack | 28.370 ± 4.398 | 36.026 ± 3.698 | 22.730 ± 4.672 | 37.080 ± 6.708 | 45.012 ± 7.965 | 27951.000 ± 30776.558 | 42913.667 ± 9380.843 | 553034.667 ± 65218.886 |
| CAMELTrack | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 |
| HyperMOT | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 |

## 基线缺项记录

- CAMELTrack。发布检查点缺可核验的训练序列清单。本协议的开发集公平性未通过。 [作者来源](https://github.com/TrackingLaboratory/CAMELTrack#-reproducing-the-paper-results-two-stage-training)
- HyperMOT。作者公开仓库未提供关联所需训练检查点。 [作者来源](https://github.com/SkyeSong38/HyperMOT/releases)

## 主表配置与训练来源

MOT17 和 MOT20 的每行共用 D-MOT 检测框与分数。D-MOT 由 COCO 初始化，在 MOT17-02、04、05、09 上微调检测头。DanceTrack 每行共用 D-Dance，由 COCO 初始化，在 DanceTrack 官方训练集训练。MOT20 复用 MOT17 的模型和参数。

| 方法 | 外观模型训练来源 | 跟踪关联训练来源 | 推理与输出 |
|---|---|---|---|
| ByteTrack、OC-SORT | 无 | 无额外学习权重 | 在线跟踪 |
| BoT-SORT、TrackTrack | 冻结 Market-1501 SBS-S50 | 无额外学习权重 | 在线跟踪与相邻帧相机补偿 |
| CGHTrack | 无 | MOT 域使用 MOT17-02、04、05、09，Dance 域使用 DanceTrack 官方训练集 | 在线跟踪 |

所有实测行均禁用离线插值与轨迹连接。TrackTrack 采用作者关联核心的单缓存适配。

## DanceTrack val · 开发验证（25 条视频）

| 方法 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | CLR_FP | CLR_FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack | 44.673 | 69.327 | 28.919 | 49.003 | 86.318 | 2066.000 | 8336.000 | 20402.000 |
| OC-SORT | 50.940 | 74.769 | 34.839 | 51.619 | 85.083 | 1873.000 | 2051.000 | 29661.000 |
| BoT-SORT | 52.502 | 75.836 | 36.494 | 54.606 | 87.938 | 1961.000 | 5831.000 | 19366.000 |
| TrackTrack | 55.114 | 76.134 | 40.134 | 57.224 | 87.708 | 1665.000 | 7819.000 | 18192.000 |
| CGHTrack | 36.734 ± 4.644 | 61.374 ± 6.753 | 22.104 ± 3.171 | 38.225 ± 4.869 | 72.547 ± 10.919 | 7692.000 ± 4328.811 | 6151.000 ± 2029.030 | 47967.667 ± 23388.320 |
| CAMELTrack | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 |
| HyperMOT | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 | 缺项 |

## DanceTrack official test

经用户确认暂缓官方计分。七组轨迹包均含 35 个序列文件，并通过内容与摘要核验。

| 方法或种子 | 覆盖 | CodaBench 分数 | 包 |
|---|---:|---|---|
| ByteTrack | 35 | 暂缓 | [ByteTrack.zip](../p6-DanceTrack-final/submission-zips/ByteTrack.zip) |
| OC-SORT | 35 | 暂缓 | [OC-SORT.zip](../p6-DanceTrack-final/submission-zips/OC-SORT.zip) |
| BoT-SORT | 35 | 暂缓 | [BoT-SORT.zip](../p6-DanceTrack-final/submission-zips/BoT-SORT.zip) |
| TrackTrack | 35 | 暂缓 | [TrackTrack.zip](../p6-DanceTrack-final/submission-zips/TrackTrack.zip) |
| CGHTrack-s0 | 35 | 暂缓 | [CGHTrack-s0.zip](../p6-DanceTrack-final/submission-zips/CGHTrack-s0.zip) |
| CGHTrack-s1 | 35 | 暂缓 | [CGHTrack-s1.zip](../p6-DanceTrack-final/submission-zips/CGHTrack-s1.zip) |
| CGHTrack-s2 | 35 | 暂缓 | [CGHTrack-s2.zip](../p6-DanceTrack-final/submission-zips/CGHTrack-s2.zip) |

登录后进入 My Submissions，选择 Test Phase 并上传一个 ZIP。首包完成后保存提交编号与完整分数，再依次上传其余已冻结的包。[竞赛入口](https://www.codabench.org/competitions/14885/) · [官方提交说明](https://docs.codabench.org/v1.34/Participants/User_Participating-in-a-Competition/)

## CGHTrack 各种子的实测分数

| 评价域 | 种子 | HOTA | AssA | IDF1 |
|---|---:|---:|---:|---:|
| MOT17 | 0 | 35.027 | 36.497 | 38.888 |
| MOT17 | 1 | 40.130 | 44.075 | 46.500 |
| MOT17 | 2 | 39.166 | 42.828 | 44.394 |
| MOT20 | 0 | 23.524 | 17.720 | 29.836 |
| MOT20 | 1 | 32.107 | 26.967 | 43.078 |
| MOT20 | 2 | 29.480 | 23.504 | 38.326 |
| DanceTrack val | 0 | 36.224 | 21.575 | 37.182 |
| DanceTrack val | 1 | 32.366 | 19.231 | 33.963 |
| DanceTrack val | 2 | 41.611 | 25.506 | 43.531 |

## E7 竞争组统计

| 种子 | 竞争组实例 | ≥9 条轨迹 | 大组占比 | 回退组数 | 回退率 |
|---|---:|---:|---:|---:|---:|
| CGHTrack-s0 | 947020 | 22278 | 2.352% | 0 | 0.000% |
| CGHTrack-s1 | 909576 | 9405 | 1.034% | 0 | 0.000% |
| CGHTrack-s2 | 923942 | 9668 | 1.046% | 0 | 0.000% |

## E8 案例

比较了 ByteTrack、TrackTrack 和 CGHTrack 种子 0。候选事件共 679 条。入选改进 3 例，失败 3 例。
