# P2 · OC-SORT 同检测输入验证

2026-11-18。验收运行 `p2-ocsort-001`。P0–P7 已完成 2/8。P2 仍在进行，MOT17-10 开发接入已完成 2/6 个外部基线（ByteTrack、OC-SORT）。

## 输入、处理、输出

- 输入为 `runs/p2-002/detections/MOT17-10-FRCNN.txt`，同一份 YOLOX-X COCO 预训练权重生成的 654 帧、18,440 条检测候选。开跑前核验 SHA-256 `00f0107933cac67d8ff0bc4667384cee6d1e478abb2d1e7dda409c7b98a4bda2`，与 ByteTrack 日志一致。
- 使用 [OC-SORT 作者实现](https://github.com/noahcao/OC_SORT) `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`。在 CPU 逐帧执行原作者 `OCSort.update`，不调用仓库附带的检测、外观特征或离线处理模块。采用作者评测入口默认配置，检测分数阈值 0.6，匹配 IoU 阈值 0.3，最大未更新帧数 30，`min_hits=3`、`delta_t=3`、`inertia=0.2`、`use_byte=False`。按作者评测入口过滤面积不大于 10 像素² 或宽高比大于 1.6 的输出框。补装 `filterpy==1.4.5`。
- 输出 5,866 条预测轨迹记录。沿用 TrackEval 1.3.0 的 MOT17 行人预处理和评分设置。预处理后计入 5,591 条预测框。预测文件帧号、框尺寸和帧内 ID 唯一性均通过检查。OC-SORT 复跑的轨迹与评测摘要逐字节相同，复用脚本后 ByteTrack 的轨迹与摘要也保持一致。

## 开发评测结果

| 方法 | HOTA ↑ | DetA ↑ | AssA ↑ | IDF1 ↑ | IDSW ↓ | 漏配真值框 ↓ | 多余预测框 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteTrack | 37.233 | 35.817 | 38.797 | 47.285 | 58 | 6,792 | 528 |
| OC-SORT | 35.064 | 31.939 | 38.585 | 43.483 | 57 | 7,580 | 332 |

前四项采用 TrackEval 的 0–100 显示。后三项为 CLEAR 评测事件或框的计数。相同检测输入下，两个跟踪器各自筛选和输出的框不同。OC-SORT 在此配置下少 196 个多余预测框，同时多 788 个未匹配的真值框，DetA 低 3.878 个百分点。少一次 IDSW 与较低的 IDF1 同时出现，说明需要结合匹配覆盖读身份指标。两套方法采用各自参数，这次对比仅覆盖对应的整体跟踪流程和这一条开发序列。

第 200 帧近处人物的检测置信度为 0.92611229。ByteTrack 输出 `ID 1`，OC-SORT 输出 `ID 30`。两个编号属于各自运行，身份是否保持需在各自的连续帧内核对。

## 产物与复跑

- `runs/p2-ocsort-001/run-config.json` 为算法版本、输入摘要和参数。
- `runs/p2-ocsort-001/trackers/OC-SORT/data/MOT17-10-FRCNN.txt` 为预测轨迹。
- `runs/p2-ocsort-001.log` 为评分原始日志。
- `runs/p2-ocsort-001/frame-0200-detection-vs-tracking.png` 为输入分数与轨迹 ID 对照。

```powershell
.\.venv\Scripts\python.exe -m pip install filterpy==1.4.5
.\.venv\Scripts\python.exe -u scripts/score_ocsort_dev.py runs/p2-ocsort-review *> runs/p2-ocsort-review.log
```

下一验收点为接入 BoT-SORT，先核对外观特征权重的训练来源，再在同一个检测缓存上评测。待用户验收后执行。
