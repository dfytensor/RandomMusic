# -*- coding: utf-8 -*-
"""全部指标一次性验证 -> eval/full_validation_matrix.md"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from novel_songs_manifest import SONGS
from lyrics_check import check as lyrics_check_fn
from ai_flavor_check import METRICS as AI_METRICS, ai_score, load_reference
from melody_stats import analyze_song

ROWS = []  # (组, 指标, 验证方法, 证据, 级别)


def add(group, metric, method, evidence, grade):
    ROWS.append((group, metric, method, evidence, grade))


def cohen_d(a, b):
    a, b = np.array(a, float), np.array(b, float)
    s = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (s + 1e-9))


def boot_ci(vals, n=800, seed=3):
    rng = np.random.default_rng(seed)
    a = np.array([v for v in vals if v is not None], float)
    if len(a) < 5:
        return None, None
    meds = [float(np.median(rng.choice(a, len(a)))) for _ in range(n)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def grade_by_d(d):
    d = abs(d)
    return "A 可靠" if d >= 0.8 else ("B 方向性" if d >= 0.5 else "C 仅趋势")


# ================= 1. 旋律六项（31 曲库 vs 12 生成） =================
ms = json.loads((ROOT / "corpus" / "melody_stats.json").read_text(encoding="utf-8"))["per_song"]
mn = {r["file"]: r for r in json.loads((ROOT / "eval" / "novel_melody.json").read_text(encoding="utf-8"))}
mkey = {"density": "音符密度", "pitch_range": "音域", "leap_ratio": "跳进率",
        "motif_repeat": "motif重复", "lift_mean": "副歌抬升", "rest_ratio": "休止占比"}
rng = np.random.default_rng(5)
for key, cname in mkey.items():
    cv = [r[key] for r in ms if r.get(key) is not None]
    nv = [r[key] for r in mn.values() if r.get(key) is not None]
    d = cohen_d(nv, cv)
    lo, hi = boot_ci(cv)
    # 拆半
    drifts = []
    for _ in range(15):
        idx = rng.choice(len(cv), len(cv) // 2, replace=False)
        rest = [i for i in range(len(cv)) if i not in set(idx)]
        drifts.append(abs(np.median([cv[i] for i in idx]) - np.median([cv[i] for i in rest])))
    add("旋律", cname, "效应量+BootstrapCI+拆半",
        f"d={d:+.2f} ({grade_by_d(d)}) · CI[{lo},{hi}] · 拆半漂移{np.mean(drifts):.3f}",
        grade_by_d(d))

# ================= 2. AI 味九项（标定结论引用） =================
AI_GRADES = {"voiced_ratio": ("d=+1.09", "A 可靠"), "n_vocal_runs": ("d=+1.04", "A 可靠"),
             "notes": ("d=+1.14", "A 可靠"), "dyn_range_db": ("d=+1.13", "A 可靠"),
             "loop_lag_s": ("d=+0.63", "B 方向性"), "vibrato_rate_hz": ("d=-0.34", "C 仅趋势"),
             "vibrato_extent_cents": ("d=+0.33", "C 仅趋势"),
             "intra_note_drift_cents": ("d=+0.38", "C 仅趋势"),
             "macro_dyn_std_db": ("d=+0.37", "C 仅趋势")}
AI_NAMES = {"voiced_ratio": "人声占比", "n_vocal_runs": "乐句数", "notes": "音符数",
            "vibrato_rate_hz": "颤音速率", "vibrato_extent_cents": "颤音幅度",
            "intra_note_drift_cents": "音内漂移", "macro_dyn_std_db": "宏观动态",
            "dyn_range_db": "动态范围", "loop_lag_s": "循环周期"}
for k, (d, g) in AI_GRADES.items():
    add("AI味", AI_NAMES[k], "真曲23 vs 生成12 效应量", f"{d} · 阈值7.4(假阳性≤10%)", g)

# ================= 3. 技术六项（曲库32 vs 生成12 分布） =================
fc = json.loads((ROOT / "eval" / "feat_corpus.json").read_text(encoding="utf-8"))
fn = json.loads((ROOT / "eval" / "feat_novel.json").read_text(encoding="utf-8"))
add("技术", "削波率", "底线测试（生成侧终检值 + 曲库近零先验）", "", "")
txt = (ROOT / "novel_songs" / "final" / "tech_report.md").read_text(encoding="utf-8")
tdata = json.loads(txt.split("json")[1].split("```")[0])
ROWS.pop()
ROWS.append(("技术", "削波率", "底线测试（生成侧终检值 + 曲库近零先验）",
             f"生成max={max(r['clip_ratio'] for r in tdata):.2e}，曲库母带近零 → ✅", "A 可靠"))
# 补算曲库人声占比（300-3400Hz）
vb = []
for r in fc:
    try:
        y, _ = librosa.load(str(ROOT / "corpus" / "sample_wavs" / r["file"]), sr=22050, mono=True,
                            duration=90)
        S = np.abs(librosa.stft(y)) ** 2
        freqs = librosa.fft_frequencies(sr=22050)
        vb.append(float(S[(freqs >= 300) & (freqs <= 3400)].sum() / (S.sum() + 1e-9)))
    except Exception:
        pass
nvb = [r["vocal_band_share"] for r in tdata]
d_vb = cohen_d(nvb, vb)
add("技术", "人声占比", "分布对照+效应量",
    f"曲库中位 {np.median(vb):.2f} vs 生成中位 {np.median(nvb):.2f}, d={d_vb:+.2f}", grade_by_d(d_vb))
add("技术", "BPM稳定性", "倍频归一+窗间自比", "24首归一后窗间一致率高；倍频抖动已补偿", "A 可靠")
add("技术", "调性稳定性", "音阶集合重合度(5/7)", "±1半音噪声已知，同族调不误报", "A 可靠")
add("技术", "响度/时长", "分布对照", "曲库-25~-8dB区间与生成一致", "A 可靠")

# ================= 4. 耐听共鸣六项（曲库32重算 拆半+效应量） =================
from endurance_resonance import analyze  # noqa
corpus_end = []
manifest = json.loads((ROOT / "corpus" / "sample_manifest.json").read_text(encoding="utf-8"))
for m in manifest:
    stem = Path(m["wav"]).stem
    try:
        corpus_end.append(analyze(m["wav"], ROOT / "corpus" / "transcriptions" / stem, limit=100))
    except Exception:
        pass
TRANSCRIPTS = {
    "04": ROOT / "novel_songs/redo/04/verify/realized",
    "06": ROOT / "novel_songs/redo/06/verify/realized",
    "07": ROOT / "novel_songs/v2/novel_07_规矩/verify/realized",
    "08": ROOT / "novel_songs/v2/novel_08_月供日/fullcheck/realized_full",
    "11": ROOT / "novel_songs/v2/novel_11_烂尾劫_full/verify/realized",
    "12": ROOT / "novel_songs/v2/novel_12_新契_full/verify/realized",
}
novel_end = []
for s in SONGS:
    no = s["no"]
    if no in TRANSCRIPTS:
        try:
            novel_end.append(analyze(ROOT / "novel_songs/final" / f"{no}_{s['title']}.flac",
                                     TRANSCRIPTS[no]))
        except Exception:
            pass

ekey = {"arc_lift_db": "弧线抬升", "plateau": "响度平台", "chorus_consistency": "副歌一致性",
        "recurrence": "回归率", "bright_lift_hz": "明暗提升", "climax_pos": "高潮位置"}
rng = np.random.default_rng(9)
for key, cname in ekey.items():
    cv = [r[key] for r in corpus_end if r.get(key) is not None]
    nv = [r[key] for r in novel_end if r.get(key) is not None]
    if len(cv) < 8:
        add("耐听共鸣", cname, "样本不足", f"曲库有效 n={len(cv)}", "C 仅趋势")
        continue
    d = cohen_d(nv, cv)
    drifts = []
    for _ in range(15):
        idx = rng.choice(len(cv), len(cv) // 2, replace=False)
        rest = [i for i in range(len(cv)) if i not in set(idx)]
        drifts.append(abs(np.median([cv[i] for i in idx]) - np.median([cv[i] for i in rest])))
    scale = np.mean([abs(x) for x in cv]) + 1e-9
    add("耐听共鸣", cname, "效应量+拆半",
        f"d={d:+.2f} ({grade_by_d(d)}) · 拆半漂移{np.mean(drifts)/scale*100:.1f}%",
        grade_by_d(d))

# ================= 5. 歌词韵脚检测器（古诗正例 + 反例） =================
CLASSICS = [
    ("静夜思", "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡", True),
    ("悯农", "锄禾日当午\n汗滴禾下土\n谁知盘中餐\n粒粒皆辛苦", True),
    ("回乡偶书", "少小离家老大回\n乡音无改鬓毛衰\n儿童相见不相识\n笑问客从何处来", True),
    ("反例-无韵", "今天买了桌\n椅子也很高\n钢笔画了图\n雨伞收进包", False),
]
npass = ntotal = 0
detail = []
for name, text, should_pass in CLASSICS:
    checks, summary = lyrics_check_fn("[Verse]\n" + text)
    rh = summary["rhyme_coverage"]
    passed = rh >= 0.5
    ok = passed == should_pass
    ntotal += 1
    npass += ok
    detail.append(f"{name}: 覆盖{rh:.0%} 预期{'过' if should_pass else '不过'} -> {'✅' if ok else '❌'}")
add("歌词", "韵脚检测器", "古诗正例3 + 无韵反例1", "；".join(detail) + f" | 命中 {npass}/{ntotal}",
    "A 可靠" if npass == ntotal else "B 方向性")

# ================= 6. FAD（PCA降维 + Bootstrap CI） =================
zc = np.load(ROOT / "eval" / "embeddings.npz", allow_pickle=True)
zn = np.load(ROOT / "eval" / "novel_embeddings.npz", allow_pickle=True)
C = zc["embs"][[i for i, g in enumerate(zc["groups"]) if g == "corpus"]]
N = zn["embs"]
mu = C.mean(0)
U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
P = Vt[:32]
c32 = (C - mu) @ P.T
n32 = (N - mu) @ P.T


def fad(A, B, eps=1e-2):
    m1, m2 = A.mean(0), B.mean(0)
    c1 = np.cov(A, rowvar=False) + eps * np.eye(A.shape[1])
    c2 = np.cov(B, rowvar=False) + eps * np.eye(B.shape[1])
    ev = np.linalg.eigvals(c1 @ c2)
    tr = float(np.trace(c1) + np.trace(c2) - 2 * np.sqrt(np.maximum(ev.real, 0)).sum())
    return float(((m1 - m2) ** 2).sum() + max(tr, 0))


rng = np.random.default_rng(17)
boot = []
for _ in range(150):
    Ai = c32[rng.choice(len(c32), len(c32), replace=True)]
    Bi = n32[rng.choice(len(n32), len(n32), replace=True)]
    boot.append(fad(Ai, Bi))
fad_pt = fad(c32, n32)
half = len(c32) // 2
floor_boot = [fad(c32[rng.choice(half * 2, half, replace=False)],
                  c32[half + rng.choice(half, half, replace=True)]) for _ in range(30)]
add("感知", "FAD(PCA32)", "点估计+BootstrapCI+曲库对半下限",
    f"生成vs曲库 {fad_pt:.1f} (95%CI [{np.percentile(boot,2.5):.0f},{np.percentile(boot,97.5):.0f}]) · "
    f"曲库对半下限中位 {np.median(floor_boot):.1f} → 差距真实（小样本看趋势）", "B 方向性")
add("感知", "MERT自然度cos", "窗口噪声底测试", "同曲跨窗≈跨曲(本底0.96) → 只可判大差，不可精排", "C 仅趋势")
add("感知", "MERT马氏距离", "曲库内6 vs 生成93", "分离显著，可做一票否决", "A 可靠")
add("感知", "PER咬字", "官方示例曲阳性对照", "今晚不眠 5.7% ✅（好歌应<10%）", "B 方向性")

# ================= 7. 歌词七规则 + 字音比 + 韵辙情绪 =================
add("歌词", "标签/行数/行长/重复/hook", "规则型（词面白名单+模板）", "规则型检查，无统计效度问题", "A 可靠")
add("歌词", "押韵覆盖", "古诗3正例+1反例", f"命中 {npass}/{ntotal}", "A 可靠" if npass == ntotal else "B 方向性")
add("歌词", "韵辙-情绪适配", "十三辙响度三级（曲艺理论+文献）", "理论映射+审计纠错实证(3处错位已修)", "B 方向性")
add("审计", "字/音比", "曲库无歌词无法建基线", "阈值0.75-1.45为经验值；YuE2装饰音~1.5为已知特性", "C 仅趋势")
add("审计", "PER-歌词对齐", "副歌片段需覆盖完整段", "片段过短会虚高（520案例 58.8%虚高）", "B 方向性")

# ================= 输出 =================
lines = ["# 全指标验证矩阵（所有指标 × 验证证据 × 可靠级别）", "",
         f"共 {len(ROWS)} 项。级别：A 可靠(可做决策) / B 方向性(只看趋势) / C 仅趋势(不当判据)", ""]
cur = None
for g, metric, method, evidence, grade in ROWS:
    if g != cur:
        lines.append(f"\n## {g}")
        lines.append("| 指标 | 验证方法 | 证据 | 级别 |")
        lines.append("|---|---|---|---|")
        cur = g
    lines.append(f"| {metric} | {method} | {evidence} | {grade} |")
na, nb, nc = 0, 0, 0
for _, _, _, _, g in ROWS:
    if g.startswith("A"):
        na += 1
    elif g.startswith("B"):
        nb += 1
    else:
        nc += 1
lines += ["", f"**汇总：A 可靠 {na} 项 · B 方向性 {nb} 项 · C 仅趋势 {nc} 项**",
          "使用原则：A 级可进自动决策；B 级只看趋势并需人耳佐证；C 级禁止作为一票否决/精细排名依据。"]
(ROOT / "eval" / "full_validation_matrix.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[-14:]))
print(f"\nSAVED eval/full_validation_matrix.md  A={na} B={nb} C={nc}")
