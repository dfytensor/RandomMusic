# v3.5 评分标准升级报告

日期: 2026-10-16 | 基线: v3.3/v3.4 (6维, eval/v3_final_report.md) | 状态: **验证通过, 已落地**

## 升级内容

11 维 = 6 旧维（orn/gap/cent/dyn/tension/micro, 副歌 30s 混音窗）
\+ 5 新维（**全曲人声 stem 口径**, 歌级恒定, art_core.py）:

| 新维 | 定义 | 人类中位 | AI冠军中位 | p (人类30 vs AI12) |
|---|---|---|---|---|
| vib_ratio | 0.8-2.5s run 中检出颤音(4-9Hz 自相关)的比例 | 0.346 | 0.215 | 0.021 |
| vib_depth | 颤音深度(去趋势 f0 残差 std, 音分) | 145.5 | 105.5 | 0.027 |
| phase_R | run onset 对伴奏拍相位集中度(1=全贴拍) | 0.222 | 0.316 | 0.062 |
| onset_rise_db | onset 前导静音→后3帧 dB 跃升(字头锐度) | 10.1 | 16.3 | 0.0006 |
| long_smooth | ≥0.8s 长音的 RMS 逐帧抖动中位(dB/frame) | 0.816 | 0.597 | 3e-5 |

来源: articulation.py~5.py 五轮分析（eval/articulation_report.md, 四条真因 p<0.05）。

## 关键决策: 阶段0 混音快验 → 混音口径弃用

articulation_mix_check.py 把 5 维改在混音副歌 30s 窗重测: **仅 1/5 保持方向**
（onset 静音测量被混音地板噪声摧毁 10dB→2dB; 乐器污染 pyin）。
→ v3.5 定稿 **全曲人声 stem 口径**（对齐已验证基准 4/5 p<0.05）。
30s 窗同样毁 vib（窗内长 run 不足）→ 新维不随窗变, 歌级恒定合并入每窗计分。
证据: eval/articulation_mix_summary.json + articulation_report.md 阶段0节。

## 基础设施

- **全库 571 Demucs 人声 stem**: human_stems/<lib_stem>/vocals.wav + beats.json
  （accomp 仅用于 beat_track 存 KB 级 beats.json, 不落盘省磁盘; 断点续跑 human_demucs.py）
- 冠军/AI seed 池 stem 回填 beats.json（album_voice/_stems, ai_stems/*, 共 31）
- art_core.py: 5 新维核心（算法逐字取自已验证 articulation2~5）
- art_scan.py: 571 首新维预热 → eval/v3_art_cache.json（train/test/gold 三份缓存共享, 只算一次）
- score_v3.py: measure(path, bpm_path, vocals_path, beats_path); 无 stem → 新维 None 跳过
  （抽卡热路径零成本）; scan() 对旧缓存条目只补新维, 全部向后兼容
- 人类库=前100s裁剪 vs AI=全曲的长度差: R 估计偏置方向对 AI 不利（保守）, 已知限制

## 验证链结果

| 检验 | 结果 | 判定 |
|---|---|---|
| 共识金标复验 (validate_v35.py, 11维5窗) | 经典 74.2 (n=25) vs 普通 71.0 (n=17), p=**0.0479** | **PASS** (门槛<0.05; v3.3 基线 0.0075) |
| test 分布 | 中位 74.5 (v3: 74.8), P25=66.3 P75=79.9 | 稳定 |
| 6维式 vs 11维 (test, Spearman) | ρ=0.760 | 高相关但有信息增量 |
| 尾部崩分 | 6/264 (2.3%) 歌因单维真离谱崩负 (如 vib_depth 472c) | 符合 v3 越界扣负设计 |
| 12 冠军排名 v3.3 vs v3.5 | ρ=0.951 | 稳定; 新序 09>10>06>08>05>11>01>03>07>02>12>04 |
| 背离修正 (n=30 冠军+seed) | onset_rise 与总分: +0.396(p=0.03) → +0.310(p=0.096 ns) | v3 奖赏"字头猛"的背离被消除显著性 |
| vib/phase 盲区 | 已纳入计分, 冠军新维 z ≤1.1 (带内温和惩罚) | 部分覆盖(组间分布差≠逐首重罚) |

## 已知稀释与限制

1. **共识 p 稀释 0.0075→0.0479**: 5 新维在人类共识组内不区分经典/普通（逐维 p>0.5）,
   等权平均稀释了 6 维的区分度。新维靶子是 AI 机器味（human30 vs AI12 验证）,
   不是人类内部好坏排序——PASS 即达标, 不做 post-hoc 调权（过拟合 42 首小样本）。
2. **冠军绝对分抬升** (5窗中位 19.2→43.0): 冠军新维多落人类带内 → 5/11 维权重"中高分"
   稀释 6 维低分。跨标准分数不可比, 排位只在同标准内有效; 对人类差距仍在
   (经典 74.2)。若未来要块级可比, 考虑块归一化/调和均值（须重新走验证链）。
3. **两阶段协议**: v3.5 终评需人声 stem → 抽卡/重扫热路径仍用 v3 6维筛,
   对 top 候选 Demucs 后 v3.5 终评（分离 ~3-5s/首, 24G 单卡可行）。
4. 共识组 3 首缺库（没有你在我有多难熬/向往/双侠）, n 沿用 v3.3 的 25/17。

## 两阶段重扫结果（rescan_v35_stage2.py, 106 候选, eval/v35_rescan_report.md）

11 池现任 v3.5 均值 **60.8 → 重选 76.0 (Δ+15.2)**, 8/11 池换冠军:
01→seed90125(+28.9) / 02→seed7776(+34.8) / 03→seed61000(+11.0) / 04→seed90403(+41.3,
与 v3 报告埋没冠军 seed90400 并列 74 实为平手) / 05→seed1110(+21.9) / 06→seed47000(+0.2 平手)
/ 09→seed90907(+3.0) / 10→seed91001(+7.6) / 12→seed91207(+18.5); 07_adj/11 不变。
v3.5 冠军分聚在 71-80 带（新维带内温和惩罚压缩了区分度, 与已知稀释#2 一致）。

## 文件索引

- 代码: art_core.py, art_scan.py, human_demucs.py(改), score_v3.py(v3.5),
  validate_v35.py, score_champions_v35.py, articulation_mix_check.py,
  consensus_dim_check.py, sanity_v35.py, diag_v35_drops.py, divergence_check_v35.py
- 数据: eval/v3_art_cache.json (571新维), eval/v3_validation.json (11维norms,
  version=v3.5), eval/v35_consensus.json, eval/v35_window_scores.json,
  eval/v35_champions.json, eval/articulation_mix_{cache,summary}.json
- 文档: SKILL.md v3.5节, articulation_report.md 阶段0节, yue_feedback.md 建议9
