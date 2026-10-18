# -*- coding: utf-8 -*-
"""三重涌现：歌词 Prosody × 演唱表现 × 声学特征，抽卡选三高 seed
目标：一个 seed 同时在三个维度达到高水平
"""
import json, sys, warnings, subprocess as sp
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from vocal_expression import vocal_expression_metrics

SR = 22050
OUT = ROOT / "novel_songs" / "triple_emergence"
OUT.mkdir(parents=True, exist_ok=True)

# ===== 三重目标（曲库中位数） =====
TARGETS = dict(
    orn_density=0.09,      # 装饰密度（越低越真人）
    gap_cv=1.38,           # 气息自然度（越接近越好）
    centroid=2135,         # 质心
    dyn=14.6,              # 动态范围
)

# ===== 最优 prompt（结合克制唱腔 + 自然气息） =====
STYLE = ("female vocal, restrained singing, minimal vocal ornamentation, "
         "clean straight tones, natural irregular breathing, warm dark tone, "
         "sparse piano with silences, breathing space between phrases, "
         "intimate, less is more, subtle emotional depth")

# 08 月供日 Prosody 版歌词（已验证 7/7）
LYR_FILE = ROOT / "novel_songs/v2/yuegongri_prosody/lyrics_v2.txt"
LYRICS = LYR_FILE.read_text(encoding="utf-8") if LYR_FILE.exists() else (
    "[verse]\n月光洒在窗台\n照见未还的账\n[chorus]\n欠天地一笔\n月供日再算")

SEEDS = [91101, 91102, 91103, 91104, 91105, 91106, 91107, 91108, 91109, 91110, 91111, 91112]

# ===== 抽卡 =====
from yue2 import YuE2Pipeline
pipe = YuE2Pipeline.from_pretrained(str(ROOT/"models"/"YuE2-3B"),
                                     vae=str(ROOT/"models"/"YuE2-Vae"),
                                     device="cuda", progress=False)
try:
    for seed in SEEDS:
        out = OUT / f"seed{seed}.flac"
        if out.exists():
            print(f"  seed{seed}: 已有", flush=True); continue
        try:
            song = pipe(style=STYLE, lyrics=LYRICS, cot="full", seed=seed)
            song.save(str(out))
            print(f"  seed{seed}: ✅ saved", flush=True)
        except Exception as e:
            print(f"  seed{seed}: FAIL {str(e)[:40]}", flush=True)
finally:
    pipe.close()

# ===== 三维评分 =====
def triple_score(f):
    """返回 (演唱分, 声学分, 综合分) 各 0-100"""
    y, _ = librosa.load(str(f), sr=SR, mono=True, duration=60)
    # 演唱维度
    v = vocal_expression_metrics(y, SR)
    orn = v.get("orn_density")
    gap = v.get("gap_cv")
    vocal = 0.0
    if orn is not None:
        vocal += max(0, 1 - abs(orn - 0.09)/0.5) * 50    # 装饰越少越好
    if gap is not None:
        vocal += max(0, 1 - abs(gap - 1.38)/1.0) * 50    # 气息越接近越好
    # 声学维度
    cent = float(librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean())
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    dyn = float(np.percentile(db,95) - np.percentile(db,10))
    acoustic = 0.0
    acoustic += max(0, 1 - abs(cent - 2135)/1200) * 50
    acoustic += max(0, 1 - abs(dyn - 14.6)/8) * 50
    return dict(vocal=round(vocal,1), acoustic=round(acoustic,1),
                total=round((vocal+acoustic)/2,1),
                orn=orn, gap=gap, cent=round(cent), dyn=round(dyn,1))

lines = ["# 三重涌现：歌词×演唱×声学 抽卡结果", "",
         "| Seed | 演唱分 | 声学分 | 综合 | 装饰/s | 气息CV | 质心Hz | 动态dB |",
         "|---|---|---|---|---|---|---|---|"]
results = []
for seed in SEEDS:
    f = OUT / f"seed{seed}.flac"
    if not f.exists(): continue
    try:
        r = triple_score(f)
        r["seed"] = seed
        results.append(r)
        lines.append(f"| {seed} | {r['vocal']} | {r['acoustic']} | **{r['total']}** "
                     f"| {r['orn']} | {r['gap']} | {r['cent']} | {r['dyn']} |")
    except Exception as e:
        lines.append(f"| {seed} | FAIL | | | | | | |")

if results:
    best = max(results, key=lambda x: x["total"])
    lines += ["", f"## 🏆 三重涌现最优: seed {best['seed']} (综合 {best['total']})"]
    lines += ["", "目标参考: 装饰≤0.15 · 气息CV≈1.38 · 质心≈2135 · 动态≈14.6"]
    (OUT / "best_seed.txt").write_text(str(best["seed"]), encoding="utf-8")

(ROOT/"eval"/"triple_emergence.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
