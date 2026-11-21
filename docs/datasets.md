# 数据下载与环境恢复

公开数据文件由各发布者分发。仓库中的清单提供官方地址、文件指纹、选取视频、标注时间线和处理规则。

| 数据 | 下载入口 | 本项目用途 |
|---|---|---|
| PersonPath22 | [发布方项目](https://github.com/amazon-science/tracking-dataset) · [数据门户及许可](https://registry.opendata.aws/person-path-22/) | 主实验。人工原生关键帧标称 5 Hz，使用发布方 FCOS 检测。 |
| DanceTrack | [作者仓库](https://github.com/DanceTrack/DanceTrack) · [作者镜像](https://huggingface.co/datasets/noahcao/dancetrack) | 25 条 val 探索；40 条 train 用于跨数据集迁移。 |
| BDD100K MOT | [下载说明](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst) · [数据许可](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst) | 五视频探索性迁移。五条源视频及哈希列于 BDD 清单。 |
| MOT17、MOT20 | [MOT17](https://motchallenge.net/data/MOT17/) · [MOT20](https://motchallenge.net/data/MOT20/) | 旧项目开发、按序列留出和迁移。 |

## PersonPath22 主实验

下载以下文件到 `data/reconnect_study/downloads`，按目标文件名保存。主实验只需要标注与已发布检测，重跑主表可以使用 CPU，无需下载完整视频。

| 目标文件名 | 发布方 URL |
|---|---|
| `personpath22-anno-amodal.zip` | [anno_amodal.zip](https://tracking-dataset-eccv-2022.s3.us-east-2.amazonaws.com/dataset/annotation/anno_amodal.zip) |
| `personpath22-detection-mot.zip` | [FCOS MOT 格式输出](https://aws-cv-sci-motion-public.s3.us-west-2.amazonaws.com/PersonPath22/public_detection_fcos/mot_format.zip) |
| `personpath22-splits.json` | [splits.json](https://tracking-dataset-eccv-2022.s3.us-east-2.amazonaws.com/dataset/annotation/splits.json) |
| `personpath22-publisher-download.py` | [来源映射脚本](https://raw.githubusercontent.com/amazon-science/tracking-dataset/main/download.py) |

哈希见 [冻结清单](../runs/reconnect_study/personpath22/manifest.json)。执行以下命令准备本地输入。

```powershell
python -m reconnect_study.data_setup --personpath-downloads data/reconnect_study/downloads --personpath-output data/reconnect_study/personpath22 --out data/reconnect_study/personpath22-local-manifest.json
```

重新生成的清单使用当前机器路径。对已有策略做复跑时仍使用仓库记录的冻结清单；加载器会映射历史工作区路径并检查文件摘要，同时保留原清单哈希。详细命令见 [reproduction.md](../reconnect_study/reproduction.md)。

## 其他公开输入

- DanceTrack 帧按历史记录放在 `data/p2/dance/train` 和 `data/p2/dance/val`。复跑需用 [YOLOX-X COCO 官方权重](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth)。所用配置和检测摘要见 [DanceTrack 清单](../runs/reconnect_study/dance_coco/manifest.json)。该历史运行的检测缓存可在原机读取，重新生成时固定源码和配置。
- BDD100K 采用 [Berkeley MOT 验证图像归档](http://dl.yf.io/bdd100k/mot20/images20-track-val-1.zip) 与 [TrackEval 公开标注归档](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip) 中的 BDD100K 验证标注。按 [试跑清单](../runs/reconnect_study/bdd100k/manifest.json) 中的五个视频 ID 取标注与图像。
- MOT17、MOT20 和旧项目权重的设置见根目录 P0–P6 记录，保留官方测试、项目留出、迁移和开发的区别。
- `vendor-lock.json` 记录第三方源码 URL 和提交号。`python tools/bootstrap_vendors.py` 可以重建本地 `vendor/`。
