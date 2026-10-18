# -*- coding: utf-8 -*-
"""11 首歌批量多维度二分爬坡抽卡
每首: Phase1 stride=2 粗探 4 seed → Phase2 最优点 stride=1 爬坡 2 seed → 选双料冠军
seed 编码: 9 + 歌号 + 序号 (如 01 首: 90101, 90103, ...)
增量写入 eval/batch_multidim_report.md，可断点续跑
"""
import json, subprocess, sys, warnings
from pathlib import Path
import numpy as np
import librosa

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from vocal_expression import vocal_expression_metrics

SR = 22050
OUTBASE = ROOT / "novel_songs" / "batch_gacha"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
REPORT = ROOT / "eval" / "batch_multidim_report.md"
TGT = dict(orn=0.09, gap=1.38, cent=2135, dyn=14.6)

def score_file(f):
    y, _ = librosa.load(str(f), sr=SR, mono=True, duration=60)
    r = vocal_expression_metrics(y, SR)
    orn, gap = r.get("orn_density"), r.get("gap_cv")
    cent = float(librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean())
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    dyn = float(np.percentile(db, 95) - np.percentile(db, 10))
    vocal = ((max(0, 1-abs(orn-0.09)/0.5)*50 + max(0, 1-abs(gap-1.38)/1.0)*50)
             if orn is not None and gap is not None else 0)
    acou = max(0,1-abs(cent-2135)/1200)*50 + max(0,1-abs(dyn-14.6)/8)*50
    dist = (abs((orn or 0)-0.09)/0.5 + abs((gap or 0)-1.38)/1.0 +
            abs(cent-2135)/1200 + abs(dyn-14.6)/8)
    return dict(orn=orn, gap=gap, cent=round(cent), dyn=round(dyn,1),
                vocal=round(vocal,1), acou=round(acou,1),
                total=round((vocal+acou)/2,1), dist=round(dist,3))

def render(no, seed, out):
    if out.exists():
        return True
    try:
        r = subprocess.run([PY, str(ROOT/"render_one.py"), no, str(seed), str(out)],
                           capture_output=True, text=True, timeout=1500)
        return "OK" in (r.stdout or "")
    except Exception:
        return False

def climb(no):
    """返回 (best_seed, best_score, all_scores)"""
    base = int(f"9{no}") * 100  # 01→90100, 08→90800
    outdir = OUTBASE / no
    results = {}
    def try_seed(sd):
        out = outdir / f"seed{sd}.flac"
        if not render(no, sd, out):
            results[sd] = None
            return
        results[sd] = score_file(out)
        r = results[sd]
        print(f"  {no} seed{sd}: total={r['total']} dist={r['dist']}", flush=True)

    # Phase 1: stride=2 粗探
    p1 = [base+1, base+3, base+5, base+7]
    for sd in p1:
        try_seed(sd)
    valid = {s: r for s, r in results.items() if r}
    if not valid:
        return None, None, results
    # Phase 2: 最优点 stride=1 爬坡
    best = max(valid, key=lambda s: valid[s]["total"])
    for sd in [best-1, best+1]:
        if sd not in results:
            try_seed(sd)
    valid = {s: r for s, r in results.items() if r}
    best = max(valid, key=lambda s: (valid[s]["total"], -valid[s]["dist"]))
    return best, valid[best], results

# ===== 主循环 =====
done = {}
if REPORT.exists():
    for ln in REPORT.read_text(encoding="utf-8").splitlines():
        if ln.startswith("| ") and "→" in ln:
            try:
                no, sd = ln.split("|")[1].strip(), ln.split("→")[0].strip().split()[-1]
                done[no] = sd
            except Exception:
                pass

lines = ["# 11 首批量多维二分爬坡抽卡", "",
         "| 歌 | 冠军seed | 演唱 | 声学 | 综合 | 距目标 | orn | gapCV | cent | dyn |",
         "|---|---|---|---|---|---|---|---|---|---|"]
if done:
    lines.append(f"<!-- 已完成: {done} -->")
REPORT.write_text("\n".join(lines), encoding="utf-8")

SONGS_NO = ["01","02","03","04","05","06","07","09","10","11","12"]
for no in SONGS_NO:
    if no in done:
        print(f"{no}: 跳过(已完成 seed{done[no]})", flush=True)
        continue
    print(f"=== 歌 {no} ===", flush=True)
    best, bs, results = climb(no)
    if best is None:
        lines.append(f"| {no} | 全部FAIL | | | | | | | | |")
    else:
        lines.append(f"| {no} | {best} | {bs['vocal']} | {bs['acou']} | **{bs['total']}** "
                     f"| {bs['dist']} | {bs['orn']} | {bs['gap']} | {bs['cent']} | {bs['dyn']} |")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{no} 冠军: seed{best} {bs}", flush=True)

print("ALL DONE", flush=True)
