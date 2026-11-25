# 投稿版本的研究范围

本文（ICDIP 投稿）只使用 `reconnect_study/method.tex` 与
`reconnect_study/experiments.tex` 中的证据。以下内容保留在仓库中供追溯，
但**不进入投稿版本**，正文、表格和结论均不引用。

## 不进入投稿版本的内容

| 内容 | 位置 | 原因 |
|---|---|---|
| 有限总体风险证书（19.95% 上界、Maclaurin 不等式、betting mixture、同时覆盖） | `method.tex` 原 §A finite-panel risk bound（已删）、`certification.py`、`risk.py`、`runs/reconnect_study/personpath22/finite-certificate-result.json` | 与主表实测失败比例回答不同问题；同时是策略族上的选点校正，与正文的选点规则重复且成本更高 |
| 随机有限总体分析（91 条随机抽取、74 条补集观测） | `runs/reconnect_study/personpath22/` 下相应产物 | 依赖上述证书，单独保留会使读者混淆两个估计量 |
| 测试代码 | `reconnect_study/tests/test_certification.py`、`test_risk.py` | 对应模块未进入投稿版本；保留以维持代码完整性 |

这些文件没有被删除，`statistical_method.md` 也保留原文，以免破坏既有哈希与
复现链。需要复核时仍可按原记录运行。

## 进入投稿版本的内容

1. 官方测试子集主对照（Table 1）：`runs/reconnect_study/personpath22/test-byte.json`、`test-bot.json`
2. 等接受量对照（§Matched acceptance）：`runs/reconnect_supplement/round2/`
3. 交互项消融与 BoostTrack 外部评分（§Where the gain comes from，Table 2）：`runs/reconnect_supplement/round3/`
4. 跨域迁移、标签敏感性与耗时：见 `reconnect_study/experiment_record.md`
5. 输入与配置审计：`runs/reconnect_study/reproducibility-audit.json`、`runs/reconnect_supplement/round3/*-root-audit.json`

## 写作口径

- 所有失败比例是**观测值**，不是保证；正文只在最后一段集中说明选点偏差，
  不在每个数字后重复限定。
- 研究设定为**无外观（detection-only）**，在实验设置中作为设计选择陈述一次。
- 第三轮结论的适用范围是"这组乘积项整体、单个工作点"，不推广到其他工作点。
