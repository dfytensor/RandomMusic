---
name: music-emergence
description: >
  涌现式音乐创作完整体系。灵感→规格→多模型生成→涌现评价→择优→母带→交付。
  基于 568 首真曲实证 + 42 首 NetEase 验证 + 49 指标全量重校 + Prosody 方法论 + 演唱表现力指标。
  涌现 = 多重性 × 相互作用 × 约束 × 非线性 × 层次 × 临界性。
---

# 涌现式音乐创作

## 全流程快速上手（从零到交付 10 步）

| 步 | 做什么 | 用什么 | 工具/命令 |
|---|---|---|---|
| 1 | 灵感 → 规格 | music_map C1-C4 簇选情绪 | `music_map.md` §3 |
| 2 | 写歌词 | **`lyricist` skill**（另一技能） | Prosody 三层翻译 + 7项门禁 |
| 3 | 定 prompt | 克制唱腔组合模板（见下） | 本 skill "Prompt 控制唱法" |
| 4 | 抽卡生成 | YuE2 3-10 seed | `triple_emergence.py` |
| 5 | 三维评分 | 演唱+声学+综合 | `vocal_expression.py` |
| 6 | 择优 | 选综合最高的 seed | `gacha_select.py` |
| 7 | 旋律手术（可选）| tie_merge/split/fold | `abc_edit.py` |
| 8 | 词曲共进化（可选）| 转谱→分析→改词→重渲染 | SheetSage2 (`remediate_novel.py`) |
| 9 | 母带 | 已有母带链 | `master_fix_v2.py` |
| 10 | 终审 | **你的耳朵** | A/B 对比 |

**技能分工：歌词找 `lyricist`，生成+评价+择优找本 skill。**

## 演唱表现力：AI味 的声学根源（新发现）

| 指标 | 真曲中位 | AI 中位 | 含义 | 工具 |
|---|---|---|---|---|
| 装饰密度 orn_density | **0.09/s** | 0.40/s | AI 装饰音是真人 4.4 倍 = 合成指纹 | `vocal_expression.py` |
| 气息自然度 gap_cv | **1.38** | 1.03 | AI 呼吸太均匀，缺少不规则呼吸 | 同上 |
| 音高偏离 pitch_dev | 12.6c | 13.4c | 接近（无显著差异）| 同上 |
| 乐句动态 phrase_dyn | 5.75dB | 5.70dB | 接近（无显著差异）| 同上 |

**核心发现：AI 的差距在唱法不在旋律——过度装饰 + 呼吸太规律。**
真人歌手用更少的装饰音表达更多情感——克制比堆砌更有力量。

## Prompt 控制唱法（已实证）

| Prompt 词 | 装饰密度 | 气息CV | 效果 |
|---|---|---|---|
| baseline | 0.41 | 1.37 | 基线 |
| + "restrained singing, minimal vocal ornamentation, clean straight tones" | **0.15 ↓63%** | 1.06 | 装饰逼近真人 0.09 |
| + "raw unpolished, natural irregular breathing" | 0.93 | **1.46** | 气息超过真人 1.38 |
| + "sparse piano with silences" | 质心 -24% | 动态 -38% | 音色变暗变柔 |

**最优组合 prompt（克制+自然气息+暖音色）：**
```
restrained singing, minimal vocal ornamentation, clean straight tones,
natural irregular breathing, warm dark tone, sparse piano with silences,
breathing space between phrases, intimate, less is more
```

**Prompt 就是 YuE2 的参数面板**——YuE2 不支持 guidance_scale，但 prompt 描述词可控制质心（±542Hz）、动态（±8.7dB）、装饰密度（±63%）、气息CV（±0.4）。

## 曲库分类目标（571 首全量实证，抽卡按类打分）

**勿用全局目标**——不同类型歌的 4 维正常范围差异巨大，抽卡前先定类：

| 类 | n | BPM中位 | 抽卡目标 orn/s | gapCV | cent | dyn |
|---|---|---|---|---|---|---|
| C1 慢板 | 187 | 108 | 0.59 | 3.39 | 1996 | 7.5 |
| C2 中速亮 | 191 | 117 | 0.41 | 3.90 | 2604 | 6.9 |
| C3 中速大动态 | 62 | 117 | 0.66 | 3.25 | 2196 | 15.6 |
| C4 快板 | 131 | 144 | 0.59 | 3.55 | 2254 | 7.1 |

```python
from category_score import score_audio, classify, score_vs_category
y, sr = librosa.load(f, sr=22050, mono=True, offset=78, duration=30)  # 副歌段！
r = score_audio(y, sr)            # 4维 + bpm（与曲库同款方法 hop=512）
cat = classify(r)                 # 定类 C1-C4
sc = score_vs_category(r, cat)    # 落同类 P25~75 每维 25 分，满分 100
```

**方法论警告（血泪教训）**：
1. 测量段落必须统一（曲库用 offset=78s 30s 副歌段）——0-60s 前奏段和副歌段数字完全不同
2. pyin hop 必须统一（曲库 hop=512）——hop=256 时 gapCV 会小 ~2.5 倍、orn 小 ~1.5 倍，两套数字不可混比
3. 跨方法比较前先用同款方法重测两边

**统一方法后的真值**：曲库真曲副歌段 gapCV 中位 3.25-3.90（呼吸比之前认为的更不规则）；AI 冠军 1.29-5.21，其中 07/09/12 已超同类中位。AI 冠军落同类 P25~75 命中数：05/06/09/12 = 3/4，7/11 首 ≥2/4。

## 音乐情绪机制：为什么音乐能影响人类情绪

**6 维情绪锚点理论**（基于 Huron ITED + Larsson 轮廓理论）:

| 锚点 | 定义 | 曲库范围 | AI 差距 |
|---|---|---|---|
| 紧张-释放 | 半音跳跃/分钟 = 情绪波动密度 | 0-5.5/分 | **AI偏高 d=+1.59** ⚠️ |
| 动态微弧 | 乐句 crescendo→decrescendo 深度 | 3.2-4.3dB | AI略高 d=+0.82 |
| 时间柔度 | 节拍偏离度 = rubato 人类感 | 0.02-0.03 | AI在区间 ✅ |
| 音高轮廓 | 旋律上行/下行 = 张力/释放 | -6~+20c | AI在区间 ✅ |
| 人声温暖 | 低频能量占比 = 体温感 | 6.4-9.4 | AI略高 d=+0.78 |

**AI 与人类的核心情绪差距**：AI 不是"没有情绪"，是**过度紧张**——9 次/分大跳 vs 真人 0-3 次。真人用更少的跳跃、更大的留白控制情绪释放时机——**留白才是情绪的载体**。

**4 类情绪画像**:
- C1 慢板: 温暖 9.4 · 紧张 1.35 · 音高上行 5c
- C2 中速亮: 温暖 6.7 · 紧张 **0** · 音高上行 7.5c
- C3 大动态: 温暖 9.4 · 紧张 5.5 · 动态弧 4.3
- C4 快板: 温暖 8.3 · 紧张 **0** · 音高持平

**抽卡时必须按类比对情绪锚点**——不能只比声学 4 维。

## 情绪导向评分公式（v2）

```python
# 旧: vocal(orn+gap)×50 + acoustic(cent+dyn)×50
# 新: 情绪锚点权重 60% + 声学 40%
emotion_score = 30*(1-|tension-cat_med|/cat_range) + 30*(1-|micro_arc-cat_med|/cat_range)
acoustic_score = vocal(orn+gap) + acoustic(cent+dyn)
total = 0.6*emotion_score + 0.4*acoustic_score
```

## 评分标准 v3.5（已验证 · 权威 · score_v3.py）

```python
# 11维 = 6旧维(orn/gap/cent/dyn/tension/micro, 副歌30s混音窗)
#      + 5新维(vib_ratio/vib_depth/phase_R/onset_rise_db/long_smooth, 全曲人声stem)
# 类别: C1<100 ≤C2<120 ≤C3<135 <C4 BPM, train=307 建标 P5/med/P95
# 计分: 落 P5~P95 → 100*(1-|v-med|/scale), 越界线性扣成负; 缺维自动跳过(n<11)
# 6旧维: measure() 自适应副歌30s窗 (v3 行为不变, 旧缓存兼容)
# 5新维: art_core.art_dims(全曲人声stem, beats.json) — 歌级恒定, 不随窗变
#   无 stem 时新维=None → 得分=纯6维(v3.3式), 抽卡热路径零成本
# 判歌两阶段: ① v3 6维筛存量(便宜) → ② 对候选 Demucs 分离后 v3.5 11维终评(贵)
```

**v3.5 验证（eval/v35_report.md）**：
- 共识金标复验 PASS: 经典 n=25 中位 74.2 vs 普通 n=17 中位 71.0, p=0.0479（v3.3 基线 0.0075;
  新维在人类共识组内不区分好坏, 稀释属预期——它们的靶子是 AI 机器味）
- test 分布稳定: 中位 74.5 (v3: 74.8), v3.3式6维 vs 11维 Spearman ρ=0.760;
  仅 2.3% (6/264) 歌因单维真离谱值(如 vib_depth 472 音分)崩分, 符合 v3 越界扣负设计
- 12 冠军排名与 v3.3 ρ=0.951 (稳定), 新冠军序: 09>10>06>08>05>11>01>03>07>02>12>04
- 背离修正: onset_rise 与总分相关 +0.396(p=0.03, v3 奖赏字头猛) → +0.310(p=0.096 不显著)
- **全库 571 首 Demucs 人声 stem 已就位** (human_stems/<lib_stem>/vocals.wav + beats.json;
  仅 vocals 落盘省磁盘, 拍点来自 accomp beat_track 存 beats.json)
- 已知稀释: 冠军新维 z 全部 ≤1.1 (多数在人类带内, 温和惩罚) → 12 冠军 5窗中位分
  从 v3.3 的 19.2 升到 43.0 (5/11 维权重被"带内"分抬高); 对比人类仍差距明显
  (共识经典 74.2)。**跨标准的分数不可直接比, 排位只在同标准内有效**

## Prompt 级破 80 实验（03 池, eval/gacha03_p35_report.md, 2026-10-16）

默认 STYLE_SUFFIX 含 "minimal vocal ornamentation, clean straight tones" —— **直接压制
vibrato/装饰音, 正是 v3.5 articulation 四征扣分项**。替换为 express 后缀
(expressive singing, gentle natural vibrato on sustained notes, legato consonant
transitions, subtle rubato micro-timing) 后 6 seed 实测:
- **2/6 破 80**: seed61024=80.8 (v3=88.6), seed61021=80.0; 默认后缀 8seed 仅 1 个 75.3
- express max 80.8 vs 默认 max 75.3; mean 60.5 vs 57.6
- **但方差双峰**: express 也产出最差 2 个 (36-40, v3=8-21) → 换后缀≠必好, 需 v3.5 筛选
- 结论: **articulation 短板部分可由 prompt 解锁** (模型会颤音, 只是默认后缀不让唱);
  破 80 路径 = express 后缀 + v3.5 两阶段选卡
- 脚本: gacha_v35_03_p.py (inline style, 不走 render_one.py 默认后缀)
- **全 12 池验证 (gacha_express_all.py, 每首 6 新 seed, eval/gacha_express_report.md)**:
  6/12 超越池内现任; 新 80+ 冠军: **12=89.5 (seed120504)**, 06=84.8, 05=81.2, 03(早前批)=80.8;
  12 号单批 3 个 80+ (89.5/83.9/82.3)。08 反例: express 后缀严重劣化 (54.9 vs 78.2,
  风格不合) → express 非万灵药, 每首须 v3.5 对照验证。长歌词触发 CUDA OOM 时开
  PYTORCH_ALLOC_CONF=expandable_segments:True + 2 次重试 (06/08/12 都撞过)。

## v3 修复史（v3.0→v3.3, 仍有效）

```python
# 6维: orn / gap / cent / dyn / tension / micro
# 类别: C1<100 ≤C2<120 ≤C3<135 <C4 BPM, train=307 建标 P5/med/P95
# 计分: 落 P5~P95 → 100*(1-|v-med|/scale), scale=p95-p5, 越界线性扣成负
# 测量: 自适应副歌段 30s (能量×起音密度峰), hop=512, SR=22050
# 判歌: measure()→score_of() 副歌段单窗, 对照 test 分布 P25=65 / P50=74.8 / P75=81
# 多窗版: 平铺0/mid/end + 能量Top1 + 中段 取中位 (判整曲一致性用)
```

**验证（eval/v3_final_report.md）**：
- 共识效度 PASS：公认经典 n=25 中位 74.3 vs 普通流行 n=17 中位 67.0，U 检验 p=0.0075
- 与 emotion_score_v2 收敛一致（01 两法皆低、08 两法皆高）
- **金标 ρ≥0.4 终验不可达（三重证据）**：全量重爬 n=548 → ρ=-0.090（满分组 vs
  非满分组 p=0.97 零关系）；诚实 train/test 拆分 119 组合+ridge 天花板 ρ=0.178；
  金标 68% 饱和在 pop=100 + 历史 49 维 ML≈基线 → **热度由艺人/年代驱动，非声学
  可预测，不能做音质锚**；替代共识金标 p=0.0075 通过
- 换段重测 ρ=0.32 是音乐本身的段落差异（曲库跨窗极差中位 39.8），非测量错误；
  **分差 <15 不下结论**

**12 冠军终版（副歌段，曲库同法）**：07=81.8 / 08=83.3 / 11=78.3 / 09=77.3 / 06=75.9
→ 达人类中上；04=2.5 / 02=15.4 / 01=24.3 拖后腿；均值 56.2 vs 人类 74.8（全库 571 中位 75.8，
**5/12 冠军达人类中位线**，最强 08 位于人类约 90 百分位，见 eval/v3_final_ranking.md）。
**AI 真实短板 = 段落一致性**（对齐检验 align_check.py：同取 0/35/70 三平铺窗、同限前 100s，
排除曲库截断 confound 后仍显著）：跨窗极差中位 AI **80.7** vs 人类 **37.7**（2.1×, p=0.0019）——
副歌接近真人，其它段垮；4 维抽卡从未优化 tension/micro，dyn/micro 系统性超人类 P95。
**外推限制**：结论基于 12 首 YuE2-3B 抽卡冠军（同专辑/同 prompt 流程，332 个渲染择优），
对"YuE2 最佳实践"成立；跨模型（Suno/ACE-Step）、非冠军平均水平、其他曲风未验证。

**人耳验证模式（可复用）**：同歌多 seed 按标准排名 → 截 30s 副歌段给用户听 →
一致则标准过检；分歧则补维度。首例验证（01_v2 11 seed）：稳定轴通过（"01确实稳"），
但发现盲区=词曲呼应+音色个性；量化发现同歌 seed 换不掉音色（MFCC 距离 42.2 vs
跨歌 76.6），音色被 prompt 锁死、seed 只影响演绎。人耳反馈已写入 eval/yue_feedback.md。

**声纹替换管线（可复用）**：`python voice_album.py --ref 干声.wav [--song 08] [--steps 30]`
— Demucs htdemucs 分离 → Seed-VC（seed-vc/，44.1k f0 模型，auto_f0 自动对齐音域）
换嗓 → 重混，输出 album_voice/。12 首演示版已跑（参考=refs/demo_faye_25s.wav，
音色距离 -18.4%）。注意：主 venv 无 pip（用 uv pip install）、勿装 torchcodec、
HF 走 HF_ENDPOINT=https://hf-mirror.com。

**换嗓对评分的影响（v3.4）**：换嗓会改 v3 分（质心被拉向参考声 + 整曲 BPM 倍频漂移
→ 类别 4/12 错乱，分数中位 67.9→37.3）。修复：`measure(path, bpm_path=伴奏轨)` 整轨
伴奏测 BPM（与副歌窗/人声无关）→ 类别漂移 4→0（verify_accomp_bpm.py 验证）。
**三段解耦流程**：①seed 抽演绎（原嗓 v3 评分选冠军）→ ②声纹换嗓（后期，换"谁在唱"）
→ ③成品重评（必用 bpm_path=accomp，只作横向对比）。换嗓后分数降低≠听感变差
（分差主要来自 cent/tension 对参考声的响应），排位须在同嗓组内比较。

**适应度错配教训（04 抽卡对照实验，8+8 seed）**：旧 4 维适应度把 v3=75.9 的 seed90400
埋在批里，选了 v3=2.5 的 seed90406 当冠军（埋没率 7/8）；用 v3 六维重抽 8 个新 seed
最优仅 62.4——**换适应度重抽 < 用对的尺子重读存量**。全库 340 渲染重扫
（rescan_v3_all.py + rescan_report.py，纯评分零渲染）：10 个可比对种子池现任均值
51.2 → 重选 79.9（+28.6），04 +73.4、02_wide +60.1、01_v2 +53.8。
**升级评分体系后的第一动作 = 用新分重扫存量库重选冠军，而不是重新抽卡。**
报告: eval/v3_rescan_report.md；重扫只收 batch_gacha/*/seed*.flac（排除 hum/mastered/根目录）。
v3.5 两阶段重扫已跑（rescan_v35_stage2.py, 106 候选, eval/v35_rescan_report.md）:
11 池现任 v3.5 均值 60.8 → 重选 76.0 (Δ+15.2), 8/11 池换冠军 —
01→90125 / 02→7776 / 03→61000 / 04→90403(与 90400 平手) / 05→1110 /
06→47000(平手) / 09→90907 / 10→91001 / 12→91207; 07/11 不变。

## 抽卡策略：多维度二分爬坡（替代随机抽卡）

**随机抽卡是错的**——种子空间有局部性（相邻 seed 得分高度相似，如 91101≈91102、91104≈91105），随机采样浪费大量算力。

```python
# 正确策略：粗探 → 爬坡 → frontier 补探
# Phase 1 粗探: stride=2~8 扫已知峰值邻域，找上升坡和悬崖
PHASE1 = [91100, 91098, 91096, 91094]
# Phase 2 精搜: 对最高点 stride=1 爬坡
PHASE2 = [91095, 91097]   # 2 发就超越随机 12 发的最优
# Phase 3 frontier: 补探两峰之间/维度冠军邻域
PHASE3 = [91099, 91093]

# 多维判优（不要只看综合分）:
# 每维找冠军（装饰/气息/质心/动态），对维度冠军邻域分别爬坡
# 用帕累托前沿 + 距目标Σ 双指标选最终胜者
# dist = Σ|各维-目标|/量程  （越小越好）
```

**实证**：随机 12 seed 最高 56.4；二分爬坡 2 发破纪录（91097=58.8，17 seed 后仍是双料冠军——综合最高 + 距目标最近）。

**先撒网后爬坡（铁律）**——死磕局部邻域是错的：
- 反例：901xx 邻域 7 发最高 50 分
- 正例：大范围粗探（1024/27182/58888/65535 跨量级撒网）**第一发就命中 100 分**
- 顺序：跨量级粗探 4 发（如 1024/27182/58888/65535）→ 对最高点 ±1 爬坡 → 若是孤立尖峰即定版

**好词抬地形（实证）**——歌词品质是抽卡天花板的乘数：
- 同一首歌：v1 词（韵辙 0.59）7 发最高 75；v2 Prosody 词（韵辙 0.80、参差句）撒网第一发 100
- 换词 = 换整个种子地形，v2 需要自己的撒网，不能沿用 v1 的爬坡坑
- 但"纸面好"只是入场券：评分永远以唱出来的声学结果计（v2 系在旧坑里 orn 反而偏高）

**注意**：某些 seed 会稳定 OOM（如 91100 三次失败）——模型对特定 seed 生成病态长度，直接放弃。

## 三重涌现：歌词×演唱×声学 同时高水平

```python
# 三维评分（每维 0-100）
vocal   = 50*(1-|orn-0.09|/0.5) + 50*(1-|gap_cv-1.38|/1.0)   # 演唱
acoustic= 50*(1-|cent-2135|/1200) + 50*(1-|dyn-14.6|/8)       # 声学
total   = (vocal + acoustic) / 2

# 抽卡 → 选综合最高 seed
# 实测结果（3 seed）：三维度部分互相牵制，没有 seed 三项全达标
# 最优 seed 是"平衡妥协"，不是全面超越
# 抽卡数越多（3→10→30），平衡点越接近帕累托前沿
```

**涌现的真相：不是某一维拉满，而是在牵制中找最优平衡点。**

## 涌现六条件 → 可执行操作

### 多重性 = 大量候选
```python
# 同一首歌 × 3模型 × 4seed × 2lyrics版本 = 24 个候选
models = ["YuE2", "ACE-Step"]  # 双模型
seeds = [42, 100, 200, 300]    # 4抽卡
lyric_versions = [v1, v2]      # 2版歌词（不同韵辙/意象系统）
```

### 相互作用 = 歌词与旋律共进化
```python
# 不是独立生成再拼，是迭代：
# Round1: 生成旋律 → 分析旋律走向 → 反过来改歌词匹配旋律情绪
# Round2: 用新歌词重新生成 → 分析新旋律 → 再调歌词
# 3轮后，词曲已经互相适应
```

### 约束 = 故意限死条件
```python
# 例：副歌只能用 5 个音（主音三五度）
# 例：每句必须以名词开头
# 例：禁止使用"心""爱""梦"三个最常见的字
# 约束逼出创意——像十四行诗的格律
```

### 非线性 = prompt 替代 guidance（YuE2 实证）
```python
# YuE2 不支持 guidance_scale → 用 prompt 描述词替代
# "sparse piano with silences"     → 质心 -24%, 动态 -38%（内敛）
# "unexpected melodic turns"       → 质心 -5%, 动态 -19%（弱效应）
# "restrained, minimal ornamentation" → 装饰密度 -63%（唱法克制）
# "natural irregular breathing"    → 气息 CV +0.4（呼吸自然化）
# prompt 是唯一可用的"参数"，用好可替代 guidance_scale 80% 的功能
```

### 临界性 = YuE2 不支持 → prompt 替代
```python
# ACE-Step 有 guidance_scale 可调临界
# YuE2 用 prompt 词推到临界状态（见"非线性"节）
```

### 层次 = 词曲情绪一致
```python
# Prosody：句长参差度 = 内心散乱程度
# 韵辙级别 = 情绪强度（洪亮/柔和/细微）
# 三层翻译：故事→演唱指令→可唱语言
```

## 完整涌现管线（含演唱维度）

```python
# Round 0: 灵感 → 规格
spec = map.select_cluster(inspiration)  # C1-C4 簇
lyrics_v1 = write_lyrics(spec, template)  # 按 Prosody 写词
lyrics_pass = lyrics_check(lyrics_v1)  # 7 项门禁

# Round 1: 多重性爆发（含 prompt 变体）
PROMPTS = {
    "restrained": "restrained singing, minimal vocal ornamentation, natural irregular breathing...",
    "explosive":  "dramatic dynamics, explosive cathartic chorus...",
    "intimate":   "sparse piano, breathing space, warm dark tone...",
}
for prompt_name, style in PROMPTS.items():
    for seed in seeds:
        song = generate(model, style, lyrics, seed)
        vocal_score = vocal_expression(song)    # 装饰+气息
        acoustic_score = netease_score(song)    # 质心+动态
        triple = (vocal_score + acoustic_score) / 2
        candidates.append((song, triple))

# Round 2: 相互作用（词曲共进化）
best = max(candidates, key=lambda x: x[1])

# Round 3: 三重涌现择优
# 装饰密度 ≤0.15 + 气息CV≈1.38 + 质心≈2135 + 动态≈14.6
# 没有 seed 三项全达标 → 选综合最高的平衡点

# 终审
human_listen(final_versions)  # 你的耳朵
```

```python
# Round 0: 灵感 → 规格
spec = map.select_cluster(inspiration)  # C1-C4 簇
lyrics_v1 = write_lyrics(spec, template)  # 按 Prosody 写词
lyrics_pass = lyrics_check(lyrics_v1)  # 7 项门禁

# Round 1: 多重性爆发
for model in ["YuE2", "ACE-Step"]:
    for seed in seeds:
        for lyric_ver in [v1, v2]:
            song = generate(model, style, lyrics_ver, seed)
            score = netease_score(song)  # 9 指标
            candidates.append((song, score))

# Round 2: 相互作用（词曲共进化）
best = top_candidates[0]
melody_analysis = transcribe(best)  # SheetSage2
melody_emotion = analyze_melody_emotion(melody_analysis)
if melody_emotion != spec.emotion:
    lyrics_v2 = rewrite_lyrics_for_melody(lyrics_v1, melody_emotion)
    song = regenerate(model, style, lyrics_v2, abc=melody_abc)

# Round 3: 临界探索
for gs in [7.4, 7.6, 7.8, 8.0]:
    song = generate(model, style, lyrics, guidance_scale=gs, seed=best_seed)
    score = evaluate(song)
    if score > best_score:
        best_gs = gs  # 锁定临界点

# Round 4: 非线性手术
# 找到最弱维度 → ABC 手术 → 重渲染
# 涌现 = 某个微调意外产生了全曲质变

# 终审
human_listen(final_versions)  # 你的耳朵
```

## 环境

| 项目 | 路径 |
|---|---|
| YuE2 venv | `F:\yinyue\.venv` |
| SheetSage2 venv | `F:\yinyue\.venv-ss2` |
| ASR venv | `F:\yinyue\.venv-asr` |
| ACE-Step venv | `F:\ACE-Step\.venv` |
| MERT 权重 | `F:\yinyue\models\MERT-v2-FullSong\` |
| Qwen3-ASR | `F:\yinyue\models\Qwen3-ASR-1.7B\` |
| 评测数据 | `F:\yinyue\eval\` |
| 演唱表现力 | `F:\yinyue\vocal_expression.py` |
| 三重涌现 | `F:\yinyue\triple_emergence.py` |
| Prompt A/B | `F:\yinyue\vocal_prompt_ab.py` |
| 专辑输出 | `F:\yinyue\novel_songs\final\` |
| 完整 map | `F:\yinyue\music_map.md` |
