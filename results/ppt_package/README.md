# HMACD-ECG PPT 图包使用说明（修订版 v2）

生成日期：2026-10-25（v2：按审阅意见修订）
数据来源：冻结 checkpoint `hmacd_k32_lc0.1_ld0.05_seed42_best.pt`（2026-09 重训版）
+ 2026-07 实验记录。所有数字均为真实实验数据。

## 建议的单页 Results 排版（10 分钟面试用）

```
HMACD-ECG — Preliminary Results & Concept Analysis
Preserving performance while examining learned concepts

┌─────────────────────┬─────────────────────┐
│ 1. Performance      │ 2. Intervention     │
│ 0.9183 vs 0.9134    │ −9.2 pp             │
│ → 01 图             │ → 03 图             │
├─────────────────────┼─────────────────────┤
│ 3. Alignment        │ 4. Reliability      │
│ median R² ≈ 0.76    │ 83.4%               │
│ → 04 图             │ → 05 图             │
└─────────────────────┴─────────────────────┘
Key takeaway: Learned concepts can support prediction and interpretation,
but their physiological validity requires further evaluation.
```

若四图放入后字号过小：**优先保留 01、03、05**；04 移到 Backup Slides，
正式页只用一句话 "Median HAS(RF) ≈ 0.76" 提及。

## 图清单与一句话 caption

### 01_performance_bar_auroc_f1.png — 性能（test split）
- **caption**：HMACD-ECG preserves classification performance — only −0.005 AUROC vs the black-box ResNet1D, with slightly higher F1 (+0.007).
- **数字**：Macro AUROC 0.9183 vs 0.9134（Δ = −0.0049）；Macro F1 0.7100 vs 0.7168（Δ = +0.0068）。test split, N=2,198。
- **v2 修订**：Δ 符号统一为 HMACD − Baseline；纵轴放宽（0.5–1.0 / 0–0.9）避免截断夸大；标注 test split。

### 02_concept_card_C14_keyleads.png — 概念卡（重点导联版）【主 PPT 推荐】
- **caption**：Concept C14's top-3 activating samples all carry MI labels — key leads II and V2, with the activation peak (red line) and the encoder's 4.35 s receptive field (shaded) marked.
- **v2 修订**：由 12 导联全叠版改为 3 样本 × II/V2 两导联，标出激活峰值与感受野范围，PPT 可读。
- 12 导联全叠的旧版（concept_card_C14.png）放 Backup。

### 03_concept_intervention_C14.png — 干预（validation split）
- **caption**：Suppressing C14 drops the mean MI probability by 9.2 pp (0.677 → 0.585) on 540 MI-positive validation samples.
- **数字**：validation split, N=540, paired drop 0.092 ± 0.061。

### 04_has_scatter_rf.png — 对齐（validation split）
- **caption**：Median HAS(RF) ≈ 0.76 — model concepts are substantially predictable from traditional ECG features (range 0.51–0.88).
- **注意**：R² 只表示"传统特征能预测概念激活"，**不构成临床验证**，表述时不要越界。

### 05_spurious_concept_C16.png — 可靠性（validation split）
- **caption**：Concept C16 puts 83.4% of its activation peaks at record boundaries with weak label association (|corr| ≤ 0.38) — a concept that influences prediction may reflect measurement artifacts rather than physiology.
- **重要**：使用 2026-09 checkpoint 的 C16，**不引用旧模型 C31 的任何数字**（旧 C31：61.5% 边界、Jaccard 0.887，在当前模型上不成立）。
- 边界敏感集合 {C12, C13, C16, C17, C01, C28} 在 2026-07 与 2026-09 两次独立训练中稳定出现——结构性现象。
- **面试衔接句**（如适用）：*"If a concept is important for prediction, how can we determine whether it reflects genuine physiology or merely measurement artifacts?"*

### 06_hmacd_pipeline_clean.png — 方法页主图
- **caption**：HMACD-ECG routes all diagnostic information through K=32 learnable prototypes; each concept activation is the max cosine similarity between a prototype and the encoder's spatial features.

## Backup Slides（可选，按需）

- **02_backup_concept_card_C14.png**（12 导联全叠版概念卡）
- **07_mixing_concept_C29.png** / **08_mixing_concept_C06.png** / **09_consistent_concept_C07.png**：
  C29 的高激活样本波形差异明显（疑似 concept mixing），C07 形态更一致（对照）。
  **注意**：肉眼差异本身不足以证明语义混合——需结合特征空间相似度（C_feat）或标签关系表述。当前研究线（时间粒度）的支撑材料。

## 诚实性说明（请勿删改）

1. **Δ 符号约定**：统一为 HMACD − Baseline（AUROC −0.0049，F1 +0.0068）。
2. **split 区分**：01 为 test split；03/04/05 为 validation split；各图底部已标注。
3. **概念编号与 checkpoint**：概念编号只在同一 checkpoint 内有意义。2026-09 重训后
   部分概念含义改变（旧 C31 → 新 C16）。本图包全部使用当前 checkpoint 的数字。
4. **数据均为真实实验结果**，未模拟或创造任何波形。

## 风格

白底学术风，蓝 #2a78d6 / 橙 #eb6834 / 绿 #1baf7a（经验证分类色），
PNG ≥220 dpi，可直接拖入 16:9 PPT。
